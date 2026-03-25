"""Labels v1 builder for the 65+ first-scope medication subset."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import uuid4

import pandas as pd

from opti_med.config import DEFAULT_SERUM_CREATININE_ITEMIDS
from opti_med.data_access.artifact_schemas import (
    ENCOUNTER_MEDICATION_LABELS_V1_FIRST_SCOPE_COLUMNS,
    ENCOUNTER_MEDICATION_LABELS_V1_FIRST_SCOPE_CONTRACT_VERSION,
    validate_encounter_index_artifact,
    validate_encounter_medication_first_scope_artifact,
    validate_encounter_medication_labels_v1_first_scope_artifact,
    validate_encounter_medication_state_artifact,
    validate_medication_events_artifact,
    validate_medication_rxnorm_mapping_artifact,
)
from opti_med.data_access.encounter_medication_state import (
    apply_medication_rxnorm_mapping_to_events,
)
from opti_med.data_access.exceptions import DataLoadError
from opti_med.data_access.medication_snapshot import (
    coalesce_event_start,
    filter_active_medication_events,
)
from opti_med.data_access.provenance import dumps_json, json_ready_value, loads_json_or_none
from opti_med.features.mappings.labs import potassium_itemids, sodium_itemids
from opti_med.labels.contracts import (
    LabelProvenanceMetadata,
    LabelSourceCategory,
    LabelWindowMetadata,
    PrimaryActionLabelCategory,
)
from opti_med.medication_semantics import infer_prn_vs_scheduled


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
LABEL_DEFINITION_VERSION = "labels_v1_65plus_first_scope.1"
LABEL_WINDOW_DEFINITION_VERSION = "post_review_to_discharge.v1"
HEMODYNAMIC_SBP_LOW_THRESHOLD = 90.0
HEMODYNAMIC_HEART_RATE_HIGH_THRESHOLD = 120.0
RESPIRATORY_RATE_LOW_THRESHOLD = 8.0
O2SAT_LOW_THRESHOLD = 90.0
POTASSIUM_LOW_THRESHOLD = 3.0
POTASSIUM_HIGH_THRESHOLD = 5.5
SODIUM_LOW_THRESHOLD = 130.0
SODIUM_HIGH_THRESHOLD = 150.0
RENAL_ABSOLUTE_CREATININE_DELTA_THRESHOLD = 0.3
RENAL_RELATIVE_CREATININE_RATIO_THRESHOLD = 1.5
ACUTE_LIFE_SUSTAINING_INGREDIENTS = frozenset({"fentanyl", "midazolam"})
FUNDAMENTALLY_DIFFERENT_DEPRESCRIBING_INGREDIENTS = frozenset({"methadone"})
ACUTE_ROUTE_TOKENS = frozenset({"iv", "intravenous", "im", "intramuscular"})
DOWNWEIGHT_PRN_CLASSES = frozenset({"benzodiazepine", "opioid"})
EMPTY_EVENT_COLUMNS = [
    "medication_event_id",
    "medication_event_type",
    "_event_start",
    "_event_stop",
    "_dose_value_numeric",
    "_dose_unit_normalized",
    "_route_normalized",
    "_schedule_semantics",
    "source_home_medrecon",
]


def build_first_scope_labels(
    *,
    encounter_medication_first_scope: pd.DataFrame,
    encounter_medication_state: pd.DataFrame,
    medication_events: pd.DataFrame,
    medication_rxnorm_mapping: pd.DataFrame,
    encounter_index: pd.DataFrame,
    labevents: pd.DataFrame | None = None,
    triage: pd.DataFrame | None = None,
    vitalsign: pd.DataFrame | None = None,
    serum_creatinine_item_ids: tuple[int, ...] = DEFAULT_SERUM_CREATININE_ITEMIDS,
) -> pd.DataFrame:
    """Build Labels v1 from post-review behavior for the 65+ first-scope subset."""
    validate_encounter_medication_first_scope_artifact(encounter_medication_first_scope)
    validate_encounter_medication_state_artifact(encounter_medication_state)
    validate_medication_events_artifact(medication_events)
    validate_medication_rxnorm_mapping_artifact(medication_rxnorm_mapping)
    validate_encounter_index_artifact(encounter_index)
    if encounter_medication_first_scope.empty:
        return pd.DataFrame(columns=ENCOUNTER_MEDICATION_LABELS_V1_FIRST_SCOPE_COLUMNS)

    build_run_id = f"encounter-medication-labels-v1-{uuid4().hex[:12]}"
    aligned = _align_first_scope_inputs(
        encounter_medication_first_scope=encounter_medication_first_scope,
        encounter_medication_state=encounter_medication_state,
        encounter_index=encounter_index,
    )
    events_by_key = _prepare_medication_event_groups(
        medication_events=medication_events,
        medication_rxnorm_mapping=medication_rxnorm_mapping,
        encounter_medication_first_scope=encounter_medication_first_scope,
    )
    labs_by_hadm = _prepare_labevents_by_hadm_id(labevents)
    vitals_by_stay = _prepare_vitals_by_stay_id(
        triage=triage,
        vitalsign=vitalsign,
    )

    medication_event_build_run_ids = _distinct_text_values(
        medication_events.get("medication_event_build_run_id", pd.Series(dtype=object))
    )
    encounter_index_build_run_ids = _distinct_text_values(
        encounter_index.get("encounter_index_build_run_id", pd.Series(dtype=object))
    )
    mapping_build_run_ids = _distinct_text_values(
        medication_rxnorm_mapping.get(
            "medication_rxnorm_mapping_build_run_id",
            pd.Series(dtype=object),
        )
    )

    rows: list[dict[str, object]] = []
    for row in aligned.to_dict(orient="records"):
        review_timestamp = pd.to_datetime(row.get("review_timestamp"), errors="coerce")
        discharge_boundary = _resolved_discharge_boundary(row)
        event_group = events_by_key.get(
            (
                _string_or_none(row.get("encounter_id")) or "",
                _string_or_none(row.get("medication_standardized")) or "",
            ),
            pd.DataFrame(columns=EMPTY_EVENT_COLUMNS),
        )
        primary_action = _evaluate_primary_action(
            row=row,
            event_group=event_group,
            review_timestamp=review_timestamp,
            discharge_boundary=discharge_boundary,
        )
        auxiliary_harms = _evaluate_auxiliary_harms(
            row=row,
            review_timestamp=review_timestamp,
            discharge_boundary=discharge_boundary,
            labs_by_hadm=labs_by_hadm,
            vitals_by_stay=vitals_by_stay,
            serum_creatinine_item_ids=serum_creatinine_item_ids,
        )
        exclusion_flags = _evaluate_exclusion_flags(row)
        label_source_tables = _label_source_tables(
            row=row,
            primary_action=primary_action,
            auxiliary_harms=auxiliary_harms,
        )
        label_window = LabelWindowMetadata(
            review_timestamp=_timestamp_to_python(review_timestamp),
            action_window_start=_timestamp_to_python(review_timestamp),
            action_window_end=_timestamp_to_python(discharge_boundary),
            harm_window_start=_timestamp_to_python(review_timestamp),
            harm_window_end=_timestamp_to_python(discharge_boundary),
            discharge_timestamp=_timestamp_to_python(discharge_boundary),
            window_censored_flag=bool(primary_action["window_censored_flag"]),
            window_definition_version=LABEL_WINDOW_DEFINITION_VERSION,
        )
        source_categories = [LabelSourceCategory.CONSTRUCTED_LABEL_LOGIC]
        if primary_action["post_review_same_medication_event_count"] > 0:
            source_categories.append(LabelSourceCategory.PRESCRIPTION_CHANGE_EVIDENCE)
        if auxiliary_harms["used_lab_evidence"]:
            source_categories.append(LabelSourceCategory.LAB_EVIDENCE)
        if auxiliary_harms["used_vital_evidence"]:
            source_categories.append(LabelSourceCategory.VITAL_EVIDENCE)
        source_categories.append(LabelSourceCategory.CLINICAL_CONTEXT_EVIDENCE)
        provenance = LabelProvenanceMetadata(
            label_definition_version=LABEL_DEFINITION_VERSION,
            source_categories=tuple(_unique_preserving_order(source_categories)),
            source_tables=tuple(label_source_tables),
            notes=(
                "Primary action uses post-review medication behavior only; "
                "current rule-score outputs are not consumed as training labels."
            ),
            provenance={
                "first_scope_build_run_id": json_ready_value(
                    row.get("encounter_medication_first_scope_build_run_id")
                ),
                "state_build_run_id": json_ready_value(
                    row.get("encounter_medication_state_build_run_id")
                ),
                "encounter_index_build_run_ids": encounter_index_build_run_ids,
                "medication_event_build_run_ids": medication_event_build_run_ids,
                "medication_rxnorm_mapping_build_run_ids": mapping_build_run_ids,
                "primary_action_reason": primary_action["primary_action_label_reason"],
                "unknown_reason": primary_action["unknown_or_insufficient_evidence_reason"],
            },
        )
        rows.append(
            {
                "subject_id": row.get("subject_id"),
                "encounter_id": row.get("encounter_id"),
                "hadm_id": row.get("hadm_id"),
                "stay_id": row.get("stay_id"),
                "review_timestamp": review_timestamp,
                "review_timestamp_source": row.get("review_timestamp_source"),
                "review_time_policy_name": row.get("review_time_policy_name"),
                "review_time_validated_flag": int(
                    pd.to_numeric(
                        pd.Series([row.get("review_time_validated_flag")]),
                        errors="coerce",
                    ).fillna(0).iloc[0]
                ),
                "review_time_capped_to_discharge_flag": int(
                    pd.to_numeric(
                        pd.Series([row.get("review_time_capped_to_discharge_flag")]),
                        errors="coerce",
                    ).fillna(0).iloc[0]
                ),
                "discharge_boundary": discharge_boundary,
                "age_proxy": row.get("age_proxy"),
                "age_group": row.get("age_group"),
                "medication_standardized": row.get("medication_standardized"),
                "medication_class_standardized": row.get("medication_class_standardized"),
                "ingredient_standardized": row.get("ingredient_standardized"),
                "medication_status_at_review": row.get("medication_status_at_review"),
                "active_at_review_flag": int(
                    pd.to_numeric(
                        pd.Series([row.get("active_at_review_flag")]),
                        errors="coerce",
                    ).fillna(0).iloc[0]
                ),
                "scheduled_vs_prn": row.get("scheduled_vs_prn"),
                "dose_value": row.get("dose_value"),
                "dose_unit": row.get("dose_unit"),
                "route": row.get("route"),
                "frequency": row.get("frequency"),
                "selected_medication_event_id": row.get("selected_medication_event_id"),
                "selected_medication_event_type": row.get("selected_medication_event_type"),
                "primary_action_label": primary_action["primary_action_label"],
                "primary_action_label_reason": primary_action["primary_action_label_reason"],
                "unknown_or_insufficient_evidence_flag": primary_action[
                    "unknown_or_insufficient_evidence_flag"
                ],
                "unknown_or_insufficient_evidence_reason": primary_action[
                    "unknown_or_insufficient_evidence_reason"
                ],
                "window_censored_flag": primary_action["window_censored_flag"],
                "renal_deterioration": auxiliary_harms["renal_deterioration"],
                "hemodynamic_instability": auxiliary_harms["hemodynamic_instability"],
                "electrolyte_instability": auxiliary_harms["electrolyte_instability"],
                "oversedation_respiratory_risk": auxiliary_harms[
                    "oversedation_respiratory_risk"
                ],
                "acute_life_sustaining_medication": exclusion_flags[
                    "acute_life_sustaining_medication"
                ],
                "fundamentally_different_deprescribing_logic": exclusion_flags[
                    "fundamentally_different_deprescribing_logic"
                ],
                "excluded_from_primary_training": exclusion_flags[
                    "excluded_from_primary_training"
                ],
                "downweight_recommended": exclusion_flags["downweight_recommended"],
                "post_review_same_medication_event_count": primary_action[
                    "post_review_same_medication_event_count"
                ],
                "post_review_same_medication_order_count": primary_action[
                    "post_review_same_medication_order_count"
                ],
                "post_review_same_medication_admin_count": primary_action[
                    "post_review_same_medication_admin_count"
                ],
                "post_review_stop_evidence_flag": primary_action[
                    "post_review_stop_evidence_flag"
                ],
                "post_review_deintensification_evidence_flag": primary_action[
                    "post_review_deintensification_evidence_flag"
                ],
                "post_review_continuation_evidence_flag": primary_action[
                    "post_review_continuation_evidence_flag"
                ],
                "primary_action_evidence_json": dumps_json(primary_action["evidence"]),
                "auxiliary_harm_evidence_json": dumps_json(auxiliary_harms["evidence"]),
                "exclusion_evidence_json": dumps_json(exclusion_flags["evidence"]),
                "label_window_json": dumps_json(_json_ready_payload(label_window)),
                "label_provenance_json": dumps_json(_json_ready_payload(provenance)),
                "label_source_tables_json": dumps_json(list(label_source_tables)),
                "encounter_medication_labels_v1_build_run_id": build_run_id,
                "encounter_medication_labels_v1_contract_version": (
                    ENCOUNTER_MEDICATION_LABELS_V1_FIRST_SCOPE_CONTRACT_VERSION
                ),
            }
        )

    labels = pd.DataFrame(rows, columns=ENCOUNTER_MEDICATION_LABELS_V1_FIRST_SCOPE_COLUMNS)
    labels = labels.sort_values(
        ["subject_id", "encounter_id", "review_timestamp", "medication_standardized"],
        na_position="last",
    ).reset_index(drop=True)
    validate_encounter_medication_labels_v1_first_scope_artifact(labels)
    return labels


def write_first_scope_label_qc_report(
    *,
    encounter_medication_labels: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write the first-scope Labels v1 QC report to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_first_scope_label_qc_report(
            encounter_medication_labels=encounter_medication_labels
        ),
        encoding="utf-8",
    )
    return output_path


def build_first_scope_label_qc_report(
    *,
    encounter_medication_labels: pd.DataFrame,
) -> str:
    """Render a Markdown QC report for the first-scope Labels v1 artifact."""
    metrics = calculate_first_scope_label_qc_metrics(
        encounter_medication_labels=encounter_medication_labels
    )
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = [
        "# Labels V1 65plus First-Scope QC",
        "",
        f"Generated at: {generated_at}",
        "",
        "## Row Counts",
        "",
        f"- label rows: {metrics['row_count']:,}",
        f"- unique subjects: {metrics['unique_subject_count']:,}",
        f"- unique encounters: {metrics['unique_encounter_count']:,}",
        f"- trainable rows: {metrics['trainable_row_count']:,}",
        "",
        "## Primary Label Prevalence",
        "",
    ]
    for label_name, count in metrics["primary_label_counts"].items():
        rate = metrics["primary_label_rates"][label_name]
        lines.append(f"- {label_name}: {count:,} ({rate:.2%})")
    lines.extend(
        [
            "",
            "## Unknown And Exclusion Rates",
            "",
            (
                "- unknown_or_insufficient_evidence rows: "
                f"{metrics['unknown_row_count']:,} ({metrics['unknown_rate']:.2%})"
            ),
            f"- excluded_from_primary_training rows: {metrics['excluded_row_count']:,} ({metrics['excluded_rate']:.2%})",
            f"- window_censored rows: {metrics['window_censored_row_count']:,} ({metrics['window_censored_rate']:.2%})",
            "",
            "## Class-Wise Primary Label Distribution",
            "",
        ]
    )
    if metrics["class_label_distribution"]:
        for class_name, label_counts in metrics["class_label_distribution"].items():
            formatted = ", ".join(
                f"{label_name}={count:,}"
                for label_name, count in label_counts.items()
            ) or "none"
            lines.append(f"- {class_name}: {formatted}")
    else:
        lines.append("- no rows")
    lines.extend(
        [
            "",
            "## Auxiliary Harm Availability",
            "",
        ]
    )
    for metric_name, payload in metrics["auxiliary_harm_metrics"].items():
        lines.append(
            f"- {metric_name}: positive={payload['positive_count']:,}, "
            f"negative={payload['negative_count']:,}, missing={payload['missing_count']:,}"
        )
    return "\n".join(lines) + "\n"


def calculate_first_scope_label_qc_metrics(
    *,
    encounter_medication_labels: pd.DataFrame,
) -> dict[str, object]:
    """Calculate compact QC metrics for the first-scope Labels v1 artifact."""
    validate_encounter_medication_labels_v1_first_scope_artifact(encounter_medication_labels)
    row_count = int(len(encounter_medication_labels))
    primary_counts = {
        str(key): int(value)
        for key, value in encounter_medication_labels["primary_action_label"]
        .fillna("null")
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
        .items()
    }
    primary_rates = {
        label_name: (count / row_count) if row_count else 0.0
        for label_name, count in primary_counts.items()
    }
    unknown_flag = (
        pd.to_numeric(
            encounter_medication_labels["unknown_or_insufficient_evidence_flag"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    excluded_flag = (
        pd.to_numeric(
            encounter_medication_labels["excluded_from_primary_training"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    censored_flag = (
        pd.to_numeric(encounter_medication_labels["window_censored_flag"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    trainable_mask = (unknown_flag == 0) & (excluded_flag == 0) & (censored_flag == 0)
    class_distribution_frame = (
        encounter_medication_labels.groupby(
            ["medication_class_standardized", "primary_action_label"],
            dropna=False,
        )
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    class_distribution = {
        str(class_name): {
            str(label_name): int(count)
            for label_name, count in class_distribution_frame.loc[class_name].to_dict().items()
        }
        for class_name in class_distribution_frame.index.tolist()
    }
    auxiliary_harm_metrics: dict[str, dict[str, int]] = {}
    for column_name in [
        "renal_deterioration",
        "hemodynamic_instability",
        "electrolyte_instability",
        "oversedation_respiratory_risk",
    ]:
        values = pd.to_numeric(encounter_medication_labels[column_name], errors="coerce")
        auxiliary_harm_metrics[column_name] = {
            "positive_count": int(values.fillna(0).eq(1).sum()),
            "negative_count": int(values.fillna(0).eq(0).sum() - values.isna().sum()),
            "missing_count": int(values.isna().sum()),
        }
    return {
        "row_count": row_count,
        "unique_subject_count": int(
            encounter_medication_labels["subject_id"].nunique(dropna=True)
        ),
        "unique_encounter_count": int(
            encounter_medication_labels["encounter_id"].nunique(dropna=True)
        ),
        "trainable_row_count": int(trainable_mask.sum()),
        "primary_label_counts": primary_counts,
        "primary_label_rates": primary_rates,
        "unknown_row_count": int(unknown_flag.sum()),
        "unknown_rate": float(unknown_flag.mean()) if row_count else 0.0,
        "excluded_row_count": int(excluded_flag.sum()),
        "excluded_rate": float(excluded_flag.mean()) if row_count else 0.0,
        "window_censored_row_count": int(censored_flag.sum()),
        "window_censored_rate": float(censored_flag.mean()) if row_count else 0.0,
        "class_label_distribution": class_distribution,
        "auxiliary_harm_metrics": auxiliary_harm_metrics,
    }


def summarize_first_scope_labels(
    *,
    encounter_medication_labels: pd.DataFrame,
) -> list[str]:
    """Return compact summary lines for the first-scope Labels v1 artifact."""
    metrics = calculate_first_scope_label_qc_metrics(
        encounter_medication_labels=encounter_medication_labels
    )
    prevalence = ", ".join(
        f"{label_name}={count:,}"
        for label_name, count in metrics["primary_label_counts"].items()
    ) or "none"
    return [
        (
            f"rows={metrics['row_count']:,}, subjects={metrics['unique_subject_count']:,}, "
            f"encounters={metrics['unique_encounter_count']:,}, "
            f"trainable_rows={metrics['trainable_row_count']:,}"
        ),
        f"primary_label_counts={prevalence}",
        (
            f"unknown_rate={metrics['unknown_row_count']:,}/{metrics['row_count']:,} "
            f"({metrics['unknown_rate']:.2%}), "
            f"excluded_rate={metrics['excluded_row_count']:,}/{metrics['row_count']:,} "
            f"({metrics['excluded_rate']:.2%})"
        ),
    ]


def _align_first_scope_inputs(
    *,
    encounter_medication_first_scope: pd.DataFrame,
    encounter_medication_state: pd.DataFrame,
    encounter_index: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = [
        "subject_id",
        "encounter_id",
        "medication_standardized",
        "review_timestamp",
    ]
    state_subset = encounter_medication_state.loc[
        :,
        key_columns
        + [
            "review_time_policy_name",
            "review_time_validated_flag",
            "review_time_capped_to_discharge_flag",
            "discharge_boundary",
            "source_tables_json",
            "encounter_medication_state_build_run_id",
        ],
    ].copy()
    state_subset = state_subset.rename(
        columns={"source_tables_json": "_state_source_tables_json"}
    )
    aligned = encounter_medication_first_scope.merge(
        state_subset,
        how="left",
        on=key_columns,
        validate="one_to_one",
        indicator=True,
    )
    if not aligned["_merge"].eq("both").all():
        missing_count = int((aligned["_merge"] != "both").sum())
        raise DataLoadError(
            "Labels v1 build requires first-scope rows to align one-to-one with the "
            f"RxNorm-enriched state artifact, but found {missing_count:,} unmatched rows."
        )
    aligned = aligned.drop(columns="_merge")
    encounter_subset = encounter_index.loc[
        :,
        [
            "encounter_id",
            "encounter_end",
            "dischtime",
            "outtime",
            "source_tables_json",
            "encounter_index_build_run_id",
        ],
    ].drop_duplicates(subset=["encounter_id"])
    encounter_subset = encounter_subset.rename(
        columns={"source_tables_json": "_encounter_source_tables_json"}
    )
    aligned = aligned.merge(
        encounter_subset,
        how="left",
        on="encounter_id",
        validate="many_to_one",
    )
    if aligned["encounter_index_build_run_id"].isna().any():
        missing_count = int(aligned["encounter_index_build_run_id"].isna().sum())
        raise DataLoadError(
            "Labels v1 build requires encounter boundary context for every first-scope row, "
            f"but {missing_count:,} rows are missing encounter-index alignment."
        )
    return aligned


def _prepare_medication_event_groups(
    *,
    medication_events: pd.DataFrame,
    medication_rxnorm_mapping: pd.DataFrame,
    encounter_medication_first_scope: pd.DataFrame,
) -> dict[tuple[str, str], pd.DataFrame]:
    events = medication_events.copy()
    events["_medication_candidate_key"] = _normalized_text_series(
        events.get("medication_normalized"),
        index=events.index,
    )
    events = apply_medication_rxnorm_mapping_to_events(
        medication_events=events,
        medication_rxnorm_mapping=medication_rxnorm_mapping,
    )
    events["_event_medication_standardized"] = (
        events.get("_medication_standardized_key", pd.Series(pd.NA, index=events.index))
        .apply(_string_or_none)
    )
    events["_event_start"] = pd.to_datetime(
        events.apply(coalesce_event_start, axis=1),
        errors="coerce",
    )
    events["_event_stop"] = pd.to_datetime(events["stoptime"], errors="coerce")
    events["_dose_value_numeric"] = pd.to_numeric(events["dose_value"], errors="coerce")
    events["_dose_unit_normalized"] = events["dose_unit"].apply(_string_or_none)
    events["_route_normalized"] = events["route"].apply(_string_or_none)
    events["_schedule_semantics"] = events.apply(
        lambda row: infer_prn_vs_scheduled(
            frequency=_string_or_none(row.get("frequency")),
            status=_string_or_none(row.get("status")),
        ),
        axis=1,
    )

    requested_keys = {
        (
            _string_or_none(row.get("encounter_id")) or "",
            _string_or_none(row.get("medication_standardized")) or "",
        )
        for row in encounter_medication_first_scope.to_dict(orient="records")
    }
    prepared = events.loc[
        events.apply(
            lambda row: (
                _string_or_none(row.get("encounter_id")) or "",
                _string_or_none(row.get("_event_medication_standardized")) or "",
            )
            in requested_keys,
            axis=1,
        )
    ].copy()
    grouped: dict[tuple[str, str], pd.DataFrame] = {}
    if prepared.empty:
        return grouped
    for (encounter_id, medication_standardized), group in prepared.groupby(
        ["encounter_id", "_event_medication_standardized"],
        dropna=False,
        sort=False,
    ):
        encounter_key = _string_or_none(encounter_id)
        medication_key = _string_or_none(medication_standardized)
        if encounter_key is None or medication_key is None:
            continue
        grouped[(encounter_key, medication_key)] = (
            group.sort_values(
                ["_event_start", "_event_stop", "medication_event_id"],
                na_position="last",
            )
            .reset_index(drop=True)
        )
    return grouped


def _evaluate_primary_action(
    *,
    row: dict[str, object],
    event_group: pd.DataFrame,
    review_timestamp: pd.Timestamp,
    discharge_boundary: pd.Timestamp,
) -> dict[str, object]:
    base_result = {
        "primary_action_label": PrimaryActionLabelCategory.ACTION_UNDETERMINED.value,
        "primary_action_label_reason": "insufficient_post_review_medication_evidence",
        "unknown_or_insufficient_evidence_flag": 1,
        "unknown_or_insufficient_evidence_reason": "insufficient_post_review_medication_evidence",
        "window_censored_flag": 0,
        "post_review_same_medication_event_count": 0,
        "post_review_same_medication_order_count": 0,
        "post_review_same_medication_admin_count": 0,
        "post_review_stop_evidence_flag": 0,
        "post_review_deintensification_evidence_flag": 0,
        "post_review_continuation_evidence_flag": 0,
        "evidence": {
            "active_non_home_event_ids_at_review": [],
            "post_review_event_ids": [],
            "candidate_stop_event_ids": [],
            "candidate_stop_times": [],
            "clear_deintensification_event_ids": [],
            "post_review_order_dose_values": [],
            "ambiguity_reasons": [],
        },
    }
    if pd.isna(review_timestamp) or pd.isna(discharge_boundary) or discharge_boundary <= review_timestamp:
        base_result.update(
            {
                "primary_action_label": PrimaryActionLabelCategory.WINDOW_CENSORED.value,
                "primary_action_label_reason": "no_post_review_window_before_discharge",
                "unknown_or_insufficient_evidence_flag": 0,
                "unknown_or_insufficient_evidence_reason": None,
                "window_censored_flag": 1,
            }
        )
        return base_result
    if int(
        pd.to_numeric(pd.Series([row.get("active_at_review_flag")]), errors="coerce")
        .fillna(0)
        .iloc[0]
    ) != 1:
        base_result.update(
            {
                "primary_action_label_reason": "medication_not_active_at_review_time",
                "unknown_or_insufficient_evidence_reason": "medication_not_active_at_review_time",
            }
        )
        return base_result
    if event_group.empty:
        return base_result

    non_home_events = event_group.loc[
        pd.to_numeric(
            event_group.get("source_home_medrecon", pd.Series(0, index=event_group.index)),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        != 1
    ].copy()
    if non_home_events.empty:
        return base_result

    active_at_review = filter_active_medication_events(non_home_events, review_timestamp)
    post_review_events = _post_review_window_events(
        event_group=non_home_events,
        review_timestamp=review_timestamp,
        discharge_boundary=discharge_boundary,
    )
    post_review_orders = post_review_events.loc[
        post_review_events["medication_event_type"] == "hospital_order"
    ].copy()
    post_review_admins = post_review_events.loc[
        post_review_events["medication_event_type"].isin(["hospital_admin", "ed_pyxis"])
    ].copy()
    candidate_stop_rows = active_at_review.loc[
        active_at_review["_event_stop"].notna()
        & (active_at_review["_event_stop"] > review_timestamp)
        & (active_at_review["_event_stop"] <= discharge_boundary)
    ].copy()
    candidate_stop_rows = candidate_stop_rows.sort_values(
        ["_event_stop", "medication_event_id"],
        na_position="last",
    )
    candidate_stop_time = (
        candidate_stop_rows["_event_stop"].iloc[0] if not candidate_stop_rows.empty else pd.NaT
    )
    deintensification = _evaluate_clear_deintensification(
        row=row,
        post_review_orders=post_review_orders,
    )
    ambiguity_reasons = list(deintensification["ambiguity_reasons"])
    if pd.notna(candidate_stop_time) and not deintensification["clear"]:
        if _medication_reappears_after_time(
            event_group=non_home_events,
            event_time=candidate_stop_time,
            discharge_boundary=discharge_boundary,
        ):
            ambiguity_reasons.append("medication_reappears_after_candidate_stop")
        if _medication_active_through_discharge(
            event_group=non_home_events,
            discharge_boundary=discharge_boundary,
        ):
            ambiguity_reasons.append("same_medication_remains_active_through_discharge")
    continuation_evidence = _continuation_evidence_present(
        event_group=non_home_events,
        post_review_events=post_review_events,
        candidate_stop_time=candidate_stop_time,
        discharge_boundary=discharge_boundary,
    )
    evidence = {
        "active_non_home_event_ids_at_review": _event_id_list(active_at_review),
        "post_review_event_ids": _event_id_list(post_review_events),
        "candidate_stop_event_ids": _event_id_list(candidate_stop_rows),
        "candidate_stop_times": [
            timestamp.strftime(TIMESTAMP_FORMAT)
            for timestamp in candidate_stop_rows["_event_stop"].dropna().tolist()
        ],
        "clear_deintensification_event_ids": deintensification["event_ids"],
        "post_review_order_dose_values": [
            {
                "medication_event_id": _string_or_none(record.get("medication_event_id")),
                "dose_value_numeric": json_ready_value(record.get("_dose_value_numeric")),
                "dose_unit": _string_or_none(record.get("_dose_unit_normalized")),
                "route": _string_or_none(record.get("_route_normalized")),
            }
            for record in post_review_orders.to_dict(orient="records")
        ],
        "ambiguity_reasons": _unique_preserving_order(ambiguity_reasons),
    }
    result = {
        **base_result,
        "post_review_same_medication_event_count": int(len(post_review_events)),
        "post_review_same_medication_order_count": int(len(post_review_orders)),
        "post_review_same_medication_admin_count": int(len(post_review_admins)),
        "post_review_stop_evidence_flag": int(pd.notna(candidate_stop_time)),
        "post_review_deintensification_evidence_flag": int(deintensification["clear"]),
        "post_review_continuation_evidence_flag": int(continuation_evidence),
        "evidence": evidence,
    }
    if deintensification["clear"] and not ambiguity_reasons:
        result.update(
            {
                "primary_action_label": (
                    PrimaryActionLabelCategory.STOPPED_OR_DEINTENSIFIED_BEFORE_DISCHARGE.value
                ),
                "primary_action_label_reason": "clear_post_review_deintensification",
                "unknown_or_insufficient_evidence_flag": 0,
                "unknown_or_insufficient_evidence_reason": None,
            }
        )
        return result
    if pd.notna(candidate_stop_time) and not ambiguity_reasons:
        result.update(
            {
                "primary_action_label": (
                    PrimaryActionLabelCategory.STOPPED_OR_DEINTENSIFIED_BEFORE_DISCHARGE.value
                ),
                "primary_action_label_reason": "clear_post_review_stop_before_discharge",
                "unknown_or_insufficient_evidence_flag": 0,
                "unknown_or_insufficient_evidence_reason": None,
            }
        )
        return result
    if ambiguity_reasons:
        result.update(
            {
                "primary_action_label_reason": "ambiguous_post_review_medication_change",
                "unknown_or_insufficient_evidence_reason": "|".join(
                    _unique_preserving_order(ambiguity_reasons)
                ),
            }
        )
        return result
    if continuation_evidence:
        result.update(
            {
                "primary_action_label": (
                    PrimaryActionLabelCategory.NO_CLEAR_STOP_OR_DEINTENSIFICATION_BEFORE_DISCHARGE.value
                ),
                "primary_action_label_reason": "post_review_continuation_evidence_without_stop_or_deintensification",
                "unknown_or_insufficient_evidence_flag": 0,
                "unknown_or_insufficient_evidence_reason": None,
            }
        )
        return result
    return result


def _evaluate_clear_deintensification(
    *,
    row: dict[str, object],
    post_review_orders: pd.DataFrame,
) -> dict[str, object]:
    baseline_dose = pd.to_numeric(pd.Series([row.get("dose_value")]), errors="coerce").iloc[0]
    baseline_unit = _string_or_none(row.get("dose_unit"))
    baseline_route = _string_or_none(row.get("route"))
    review_schedule = _string_or_none(row.get("scheduled_vs_prn"))
    result = {
        "clear": False,
        "ambiguity_reasons": [],
        "event_ids": [],
    }
    if post_review_orders.empty:
        return result
    dose_numeric_orders = post_review_orders.loc[
        post_review_orders["_dose_value_numeric"].notna()
    ].copy()
    if dose_numeric_orders.empty:
        return result
    if pd.isna(baseline_dose) or baseline_unit is None or baseline_route is None:
        if (dose_numeric_orders["_dose_value_numeric"] < baseline_dose).fillna(False).any():
            result["ambiguity_reasons"].append("missing_review_time_dose_context")
        return result
    if review_schedule != "scheduled":
        lower_any = (
            dose_numeric_orders["_dose_value_numeric"] < float(baseline_dose)
        ).any()
        if lower_any:
            result["ambiguity_reasons"].append("review_or_post_review_prn_context")
        return result
    comparable = dose_numeric_orders.loc[
        (dose_numeric_orders["_dose_unit_normalized"] == baseline_unit)
        & (dose_numeric_orders["_route_normalized"] == baseline_route)
        & (dose_numeric_orders["_schedule_semantics"] == "scheduled")
    ].copy()
    lower = comparable.loc[comparable["_dose_value_numeric"] < float(baseline_dose)].copy()
    if lower.empty:
        lower_any = (dose_numeric_orders["_dose_value_numeric"] < float(baseline_dose)).any()
        if lower_any:
            result["ambiguity_reasons"].append("post_review_dose_change_not_comparable")
        return result
    if len(comparable) != len(dose_numeric_orders):
        result["ambiguity_reasons"].append("post_review_dose_change_not_comparable")
    if comparable.loc[comparable["_dose_value_numeric"] >= float(baseline_dose)].shape[0] > 0:
        result["ambiguity_reasons"].append("conflicting_post_review_dose_levels")
    distinct_lower_doses = sorted(
        {
            round(float(value), 6)
            for value in lower["_dose_value_numeric"].dropna().tolist()
        }
    )
    if len(distinct_lower_doses) != 1:
        result["ambiguity_reasons"].append("multiple_post_review_lower_dose_levels")
    if result["ambiguity_reasons"]:
        return result
    result["clear"] = True
    result["event_ids"] = _event_id_list(lower)
    return result


def _continuation_evidence_present(
    *,
    event_group: pd.DataFrame,
    post_review_events: pd.DataFrame,
    candidate_stop_time: pd.Timestamp,
    discharge_boundary: pd.Timestamp,
) -> bool:
    if _medication_active_through_discharge(
        event_group=event_group,
        discharge_boundary=discharge_boundary,
    ):
        return True
    if pd.notna(candidate_stop_time):
        return _medication_reappears_after_time(
            event_group=event_group,
            event_time=candidate_stop_time,
            discharge_boundary=discharge_boundary,
        )
    return not post_review_events.empty


def _medication_active_through_discharge(
    *,
    event_group: pd.DataFrame,
    discharge_boundary: pd.Timestamp,
) -> bool:
    orders = event_group.loc[event_group["medication_event_type"] == "hospital_order"].copy()
    if orders.empty or pd.isna(discharge_boundary):
        return False
    return bool(
        (
            orders["_event_start"].notna()
            & (orders["_event_start"] <= discharge_boundary)
            & (orders["_event_stop"].isna() | (orders["_event_stop"] > discharge_boundary))
        ).any()
    )


def _medication_reappears_after_time(
    *,
    event_group: pd.DataFrame,
    event_time: pd.Timestamp,
    discharge_boundary: pd.Timestamp,
) -> bool:
    if pd.isna(event_time):
        return False
    post_time = event_group.loc[
        event_group["_event_start"].notna()
        & (event_group["_event_start"] > event_time)
        & (event_group["_event_start"] <= discharge_boundary)
    ].copy()
    return not post_time.empty


def _post_review_window_events(
    *,
    event_group: pd.DataFrame,
    review_timestamp: pd.Timestamp,
    discharge_boundary: pd.Timestamp,
) -> pd.DataFrame:
    if event_group.empty:
        return event_group.copy()
    in_window = (
        (
            event_group["_event_start"].notna()
            & (event_group["_event_start"] > review_timestamp)
            & (event_group["_event_start"] <= discharge_boundary)
        )
        | (
            event_group["_event_stop"].notna()
            & (event_group["_event_stop"] > review_timestamp)
            & (event_group["_event_stop"] <= discharge_boundary)
        )
    )
    return event_group.loc[in_window].copy()


def _evaluate_auxiliary_harms(
    *,
    row: dict[str, object],
    review_timestamp: pd.Timestamp,
    discharge_boundary: pd.Timestamp,
    labs_by_hadm: dict[int, pd.DataFrame],
    vitals_by_stay: dict[int, pd.DataFrame],
    serum_creatinine_item_ids: tuple[int, ...],
) -> dict[str, object]:
    if pd.isna(review_timestamp) or pd.isna(discharge_boundary) or discharge_boundary <= review_timestamp:
        evidence = {
            "renal_deterioration": {"reason": "window_censored"},
            "hemodynamic_instability": {"reason": "window_censored"},
            "electrolyte_instability": {"reason": "window_censored"},
            "oversedation_respiratory_risk": {"reason": "window_censored"},
        }
        return {
            "renal_deterioration": None,
            "hemodynamic_instability": None,
            "electrolyte_instability": None,
            "oversedation_respiratory_risk": None,
            "used_lab_evidence": False,
            "used_vital_evidence": False,
            "evidence": evidence,
        }
    renal_value, renal_evidence = _evaluate_renal_deterioration(
        hadm_id=_normalized_int(row.get("hadm_id")),
        review_timestamp=review_timestamp,
        discharge_boundary=discharge_boundary,
        labs_by_hadm=labs_by_hadm,
        serum_creatinine_item_ids=serum_creatinine_item_ids,
    )
    electrolyte_value, electrolyte_evidence = _evaluate_electrolyte_instability(
        hadm_id=_normalized_int(row.get("hadm_id")),
        review_timestamp=review_timestamp,
        discharge_boundary=discharge_boundary,
        labs_by_hadm=labs_by_hadm,
    )
    hemodynamic_value, hemodynamic_evidence = _evaluate_hemodynamic_instability(
        stay_id=_normalized_int(row.get("stay_id")),
        review_timestamp=review_timestamp,
        discharge_boundary=discharge_boundary,
        vitals_by_stay=vitals_by_stay,
    )
    oversedation_value, oversedation_evidence = _evaluate_oversedation_respiratory_risk(
        stay_id=_normalized_int(row.get("stay_id")),
        review_timestamp=review_timestamp,
        discharge_boundary=discharge_boundary,
        vitals_by_stay=vitals_by_stay,
    )
    evidence = {
        "renal_deterioration": renal_evidence,
        "hemodynamic_instability": hemodynamic_evidence,
        "electrolyte_instability": electrolyte_evidence,
        "oversedation_respiratory_risk": oversedation_evidence,
    }
    return {
        "renal_deterioration": renal_value,
        "hemodynamic_instability": hemodynamic_value,
        "electrolyte_instability": electrolyte_value,
        "oversedation_respiratory_risk": oversedation_value,
        "used_lab_evidence": bool(
            renal_evidence.get("used_observations") or electrolyte_evidence.get("used_observations")
        ),
        "used_vital_evidence": bool(
            hemodynamic_evidence.get("used_observations")
            or oversedation_evidence.get("used_observations")
        ),
        "evidence": evidence,
    }


def _evaluate_renal_deterioration(
    *,
    hadm_id: int | None,
    review_timestamp: pd.Timestamp,
    discharge_boundary: pd.Timestamp,
    labs_by_hadm: dict[int, pd.DataFrame],
    serum_creatinine_item_ids: tuple[int, ...],
) -> tuple[int | None, dict[str, object]]:
    if hadm_id is None or hadm_id not in labs_by_hadm:
        return None, {"reason": "no_lab_context_for_encounter", "used_observations": 0}
    group = labs_by_hadm[hadm_id]
    creatinine = group.loc[group["itemid"].isin(serum_creatinine_item_ids)].copy()
    if creatinine.empty:
        return None, {"reason": "no_creatinine_observations", "used_observations": 0}
    baseline = creatinine.loc[creatinine["charttime"] <= review_timestamp].sort_values(
        "charttime",
        na_position="last",
    )
    post_review = creatinine.loc[
        (creatinine["charttime"] > review_timestamp)
        & (creatinine["charttime"] <= discharge_boundary)
    ].copy()
    if baseline.empty:
        return None, {"reason": "missing_pre_review_creatinine_baseline", "used_observations": 0}
    if post_review.empty:
        return None, {"reason": "no_post_review_creatinine_observations", "used_observations": 0}
    baseline_row = baseline.iloc[-1]
    baseline_value = float(baseline_row["valuenum"])
    post_review_max = float(post_review["valuenum"].max())
    positive = int(
        (post_review_max >= baseline_value + RENAL_ABSOLUTE_CREATININE_DELTA_THRESHOLD)
        or (post_review_max >= baseline_value * RENAL_RELATIVE_CREATININE_RATIO_THRESHOLD)
    )
    return positive, {
        "used_observations": int(len(post_review)),
        "baseline_time": baseline_row["charttime"].strftime(TIMESTAMP_FORMAT),
        "baseline_value": round(baseline_value, 3),
        "post_review_max": round(post_review_max, 3),
        "post_review_latest_time": post_review["charttime"].max().strftime(TIMESTAMP_FORMAT),
        "thresholds": {
            "absolute_delta_mg_dl": RENAL_ABSOLUTE_CREATININE_DELTA_THRESHOLD,
            "relative_ratio": RENAL_RELATIVE_CREATININE_RATIO_THRESHOLD,
        },
    }


def _evaluate_electrolyte_instability(
    *,
    hadm_id: int | None,
    review_timestamp: pd.Timestamp,
    discharge_boundary: pd.Timestamp,
    labs_by_hadm: dict[int, pd.DataFrame],
) -> tuple[int | None, dict[str, object]]:
    if hadm_id is None or hadm_id not in labs_by_hadm:
        return None, {"reason": "no_lab_context_for_encounter", "used_observations": 0}
    group = labs_by_hadm[hadm_id]
    post_review = group.loc[
        (group["charttime"] > review_timestamp) & (group["charttime"] <= discharge_boundary)
    ].copy()
    potassium = post_review.loc[post_review["itemid"].isin(potassium_itemids())].copy()
    sodium = post_review.loc[post_review["itemid"].isin(sodium_itemids())].copy()
    if potassium.empty and sodium.empty:
        return None, {"reason": "no_post_review_electrolyte_observations", "used_observations": 0}
    potassium_abnormal = bool(
        (potassium["valuenum"] < POTASSIUM_LOW_THRESHOLD).any()
        or (potassium["valuenum"] > POTASSIUM_HIGH_THRESHOLD).any()
    )
    sodium_abnormal = bool(
        (sodium["valuenum"] < SODIUM_LOW_THRESHOLD).any()
        or (sodium["valuenum"] > SODIUM_HIGH_THRESHOLD).any()
    )
    return int(potassium_abnormal or sodium_abnormal), {
        "used_observations": int(len(potassium) + len(sodium)),
        "potassium_min": _rounded_or_none(potassium["valuenum"].min() if not potassium.empty else None),
        "potassium_max": _rounded_or_none(potassium["valuenum"].max() if not potassium.empty else None),
        "sodium_min": _rounded_or_none(sodium["valuenum"].min() if not sodium.empty else None),
        "sodium_max": _rounded_or_none(sodium["valuenum"].max() if not sodium.empty else None),
        "thresholds": {
            "potassium_low": POTASSIUM_LOW_THRESHOLD,
            "potassium_high": POTASSIUM_HIGH_THRESHOLD,
            "sodium_low": SODIUM_LOW_THRESHOLD,
            "sodium_high": SODIUM_HIGH_THRESHOLD,
        },
    }


def _evaluate_hemodynamic_instability(
    *,
    stay_id: int | None,
    review_timestamp: pd.Timestamp,
    discharge_boundary: pd.Timestamp,
    vitals_by_stay: dict[int, pd.DataFrame],
) -> tuple[int | None, dict[str, object]]:
    if stay_id is None or stay_id not in vitals_by_stay:
        return None, {"reason": "no_vital_context_for_encounter", "used_observations": 0}
    post_review = vitals_by_stay[stay_id].loc[
        (vitals_by_stay[stay_id]["charttime"] > review_timestamp)
        & (vitals_by_stay[stay_id]["charttime"] <= discharge_boundary)
    ].copy()
    has_hemodynamic_data = post_review[["sbp", "heartrate"]].notna().any(axis=1)
    post_review = post_review.loc[has_hemodynamic_data].copy()
    if post_review.empty:
        return None, {"reason": "no_post_review_hemodynamic_observations", "used_observations": 0}
    positive = int(
        (post_review["sbp"] <= HEMODYNAMIC_SBP_LOW_THRESHOLD).any()
        or (post_review["heartrate"] >= HEMODYNAMIC_HEART_RATE_HIGH_THRESHOLD).any()
    )
    return positive, {
        "used_observations": int(len(post_review)),
        "sbp_min": _rounded_or_none(post_review["sbp"].min()),
        "heart_rate_max": _rounded_or_none(post_review["heartrate"].max()),
        "thresholds": {
            "sbp_low": HEMODYNAMIC_SBP_LOW_THRESHOLD,
            "heart_rate_high": HEMODYNAMIC_HEART_RATE_HIGH_THRESHOLD,
        },
    }


def _evaluate_oversedation_respiratory_risk(
    *,
    stay_id: int | None,
    review_timestamp: pd.Timestamp,
    discharge_boundary: pd.Timestamp,
    vitals_by_stay: dict[int, pd.DataFrame],
) -> tuple[int | None, dict[str, object]]:
    if stay_id is None or stay_id not in vitals_by_stay:
        return None, {"reason": "no_vital_context_for_encounter", "used_observations": 0}
    post_review = vitals_by_stay[stay_id].loc[
        (vitals_by_stay[stay_id]["charttime"] > review_timestamp)
        & (vitals_by_stay[stay_id]["charttime"] <= discharge_boundary)
    ].copy()
    has_respiratory_data = post_review[["resprate", "o2sat"]].notna().any(axis=1)
    post_review = post_review.loc[has_respiratory_data].copy()
    if post_review.empty:
        return None, {"reason": "no_post_review_respiratory_observations", "used_observations": 0}
    positive = int(
        (post_review["resprate"] <= RESPIRATORY_RATE_LOW_THRESHOLD).any()
        or (post_review["o2sat"] < O2SAT_LOW_THRESHOLD).any()
    )
    return positive, {
        "used_observations": int(len(post_review)),
        "respiratory_rate_min": _rounded_or_none(post_review["resprate"].min()),
        "o2sat_min": _rounded_or_none(post_review["o2sat"].min()),
        "thresholds": {
            "respiratory_rate_low": RESPIRATORY_RATE_LOW_THRESHOLD,
            "o2sat_low": O2SAT_LOW_THRESHOLD,
        },
    }


def _evaluate_exclusion_flags(row: dict[str, object]) -> dict[str, object]:
    ingredient = _string_or_none(row.get("ingredient_standardized")) or _string_or_none(
        row.get("medication_standardized")
    )
    route = _string_or_none(row.get("route"))
    selected_event_type = _string_or_none(row.get("selected_medication_event_type"))
    scheduled_vs_prn = _string_or_none(row.get("scheduled_vs_prn"))
    medication_class = _string_or_none(row.get("medication_class_standardized"))
    duration_before_review_hours = pd.to_numeric(
        pd.Series([row.get("duration_before_review_hours")]),
        errors="coerce",
    ).iloc[0]
    new_start_flag = int(
        pd.to_numeric(
            pd.Series([row.get("newly_started_during_encounter_inferred")]),
            errors="coerce",
        )
        .fillna(0)
        .iloc[0]
    )
    acute_context = bool(
        route in ACUTE_ROUTE_TOKENS
        or selected_event_type == "hospital_admin"
        or (
            new_start_flag == 1
            and pd.notna(duration_before_review_hours)
            and float(duration_before_review_hours) <= 24.0
        )
    )
    acute_flag = int(ingredient in ACUTE_LIFE_SUSTAINING_INGREDIENTS and acute_context)
    logic_flag = int(ingredient in FUNDAMENTALLY_DIFFERENT_DEPRESCRIBING_INGREDIENTS)
    excluded_flag = int(acute_flag == 1 or logic_flag == 1)
    downweight_flag = int(
        excluded_flag == 0
        and scheduled_vs_prn == "prn"
        and medication_class in DOWNWEIGHT_PRN_CLASSES
    )
    reasons: list[str] = []
    if acute_flag == 1:
        reasons.append("acute_life_sustaining_medication")
    if logic_flag == 1:
        reasons.append("fundamentally_different_deprescribing_logic")
    if downweight_flag == 1:
        reasons.append("prn_rescue_context_downweight")
    return {
        "acute_life_sustaining_medication": acute_flag,
        "fundamentally_different_deprescribing_logic": logic_flag,
        "excluded_from_primary_training": excluded_flag,
        "downweight_recommended": downweight_flag,
        "evidence": {
            "ingredient_standardized": ingredient,
            "route": route,
            "selected_medication_event_type": selected_event_type,
            "scheduled_vs_prn": scheduled_vs_prn,
            "duration_before_review_hours": _rounded_or_none(duration_before_review_hours),
            "newly_started_during_encounter_inferred": new_start_flag,
            "reasons": reasons,
        },
    }


def _label_source_tables(
    *,
    row: dict[str, object],
    primary_action: dict[str, object],
    auxiliary_harms: dict[str, object],
) -> tuple[str, ...]:
    tables = {
        "encounter_medication_first_scope_65plus",
        "encounter_medication_state_65plus_rxnorm",
        "medication_events",
        "encounter_index",
    }
    tables.update(_json_string_list(row.get("_state_source_tables_json")))
    tables.update(_json_string_list(row.get("_encounter_source_tables_json")))
    if primary_action["post_review_same_medication_event_count"] > 0:
        tables.add("medication_events")
    if auxiliary_harms["used_lab_evidence"]:
        tables.add("labevents")
    if auxiliary_harms["used_vital_evidence"]:
        tables.update({"triage", "vitalsign"})
    return tuple(sorted(tables))


def _prepare_labevents_by_hadm_id(
    labevents: pd.DataFrame | None,
) -> dict[int, pd.DataFrame]:
    if labevents is None or labevents.empty:
        return {}
    prepared = labevents.loc[:, ["hadm_id", "itemid", "charttime", "valuenum"]].copy()
    prepared["hadm_id"] = pd.to_numeric(prepared["hadm_id"], errors="coerce").astype("Int64")
    prepared["itemid"] = pd.to_numeric(prepared["itemid"], errors="coerce").astype("Int64")
    prepared["charttime"] = pd.to_datetime(prepared["charttime"], errors="coerce")
    prepared["valuenum"] = pd.to_numeric(prepared["valuenum"], errors="coerce")
    prepared = prepared.dropna(subset=["hadm_id", "itemid", "charttime", "valuenum"])
    if prepared.empty:
        return {}
    return {
        int(hadm_id): group.sort_values("charttime", na_position="last").reset_index(drop=True)
        for hadm_id, group in prepared.groupby("hadm_id", sort=False)
    }


def _prepare_vitals_by_stay_id(
    *,
    triage: pd.DataFrame | None,
    vitalsign: pd.DataFrame | None,
) -> dict[int, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    if triage is not None and not triage.empty:
        triage_time_column = "charttime" if "charttime" in triage.columns else "intime"
        triage_columns = [
            "stay_id",
            triage_time_column,
            "sbp",
            "dbp",
            "heartrate",
            "resprate",
            "o2sat",
            "pain",
        ]
        available_columns = [column for column in triage_columns if column in triage.columns]
        triage_frame = triage.loc[:, available_columns].copy()
        triage_frame = triage_frame.rename(columns={triage_time_column: "charttime"})
        frames.append(triage_frame)
    if vitalsign is not None and not vitalsign.empty:
        vital_columns = [
            column
            for column in ["stay_id", "charttime", "sbp", "dbp", "heartrate", "resprate", "o2sat", "pain"]
            if column in vitalsign.columns
        ]
        if {"stay_id", "charttime"}.issubset(vital_columns):
            frames.append(vitalsign.loc[:, vital_columns].copy())
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["stay_id"] = pd.to_numeric(combined["stay_id"], errors="coerce").astype("Int64")
    combined["charttime"] = pd.to_datetime(combined["charttime"], errors="coerce")
    for column_name in ["sbp", "dbp", "heartrate", "resprate", "o2sat", "pain"]:
        if column_name not in combined:
            combined[column_name] = pd.NA
        combined[column_name] = pd.to_numeric(combined[column_name], errors="coerce")
    combined = combined.dropna(subset=["stay_id", "charttime"])
    if combined.empty:
        return {}
    return {
        int(stay_id): group.sort_values("charttime", na_position="last").reset_index(drop=True)
        for stay_id, group in combined.groupby("stay_id", sort=False)
    }


def _resolved_discharge_boundary(row: dict[str, object]) -> pd.Timestamp:
    for column_name in ["discharge_boundary", "encounter_end", "dischtime", "outtime"]:
        candidate = pd.to_datetime(row.get(column_name), errors="coerce")
        if pd.notna(candidate):
            return candidate
    return pd.NaT


def _event_id_list(dataframe: pd.DataFrame) -> list[str]:
    return [
        event_id
        for event_id in dataframe.get("medication_event_id", pd.Series(dtype=object))
        .apply(_string_or_none)
        .tolist()
        if event_id is not None
    ]


def _normalized_text_series(
    series: pd.Series | None,
    *,
    index: pd.Index,
) -> pd.Series:
    if series is None:
        return pd.Series(pd.NA, index=index, dtype="object")
    return series.apply(_string_or_none)


def _json_string_list(value: object) -> list[str]:
    payload = loads_json_or_none(value)
    if not isinstance(payload, list):
        return []
    return sorted(
        {
            text
            for text in [_string_or_none(item) for item in payload]
            if text is not None
        }
    )


def _json_ready_payload(value: object) -> object:
    if is_dataclass(value):
        return _json_ready_payload(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_ready_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready_payload(item) for item in value]
    return json_ready_value(value)


def _distinct_text_values(series: pd.Series) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                text
                for text in series.apply(_string_or_none).tolist()
                if text is not None
            }
        )
    )


def _unique_preserving_order(values: list[object]) -> list[object]:
    seen: set[object] = set()
    ordered: list[object] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _string_or_none(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip().lower()
    return text or None


def _normalized_int(value: object) -> int | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return int(numeric)


def _timestamp_to_python(value: pd.Timestamp) -> datetime | None:
    if pd.isna(value):
        return None
    return value.to_pydatetime()


def _rounded_or_none(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return round(float(numeric), 3)
