"""Schema assertions for persisted analytical artifacts built from standardized tables."""

from __future__ import annotations

import json

import pandas as pd

from opti_med.data_access.exceptions import DataLoadError
from opti_med.medication_semantics import FIRST_SCOPE_SUPPORTED_CLASSES


ENCOUNTER_INDEX_CONTRACT_VERSION = "encounter_index.v1"
MEDICATION_EVENTS_CONTRACT_VERSION = "medication_events.v1"
ENCOUNTER_MEDICATION_STATE_CONTRACT_VERSION = "encounter_medication_state.v1"
MEDICATION_RXNORM_MAPPING_CONTRACT_VERSION = "medication_rxnorm_mapping.v1"
ENCOUNTER_MEDICATION_SEMANTICS_CONTRACT_VERSION = "encounter_medication_semantics.v1"
ENCOUNTER_MEDICATION_BURDEN_CONTRACT_VERSION = "encounter_medication_burden.v1"

ENCOUNTER_INDEX_COLUMNS = [
    "subject_id",
    "sex",
    "age_proxy",
    "age_group",
    "encounter_id",
    "hadm_id",
    "stay_id",
    "encounter_source",
    "linked_ed_stay_flag",
    "linked_hospital_admission_flag",
    "admission_type",
    "admittime",
    "dischtime",
    "hospital_length_of_stay_days",
    "intime",
    "outtime",
    "ed_length_of_stay_hours",
    "ed_disposition",
    "arrival_transport",
    "encounter_start",
    "encounter_end",
    "source_tables_json",
    "source_record_provenance_json",
    "encounter_index_build_run_id",
    "encounter_index_contract_version",
]

MEDICATION_EVENT_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "encounter_id",
    "encounter_source",
    "encounter_start",
    "encounter_end",
    "medication_event_id",
    "medication_event_type",
    "event_source_category",
    "event_source_table",
    "raw_medication_name",
    "medication_name",
    "medication_normalized",
    "medication_prestandardized_text",
    "event_time",
    "starttime",
    "stoptime",
    "route",
    "frequency",
    "status",
    "dose_value",
    "dose_unit",
    "pharmacy_id",
    "poe_id",
    "emar_id",
    "emar_seq",
    "source_home_medrecon",
    "source_ed_pyxis",
    "source_hospital_order",
    "source_hospital_admin",
    "pharmacy_enriched_flag",
    "order_enrichment_applied_flag",
    "order_enrichment_source_table",
    "continued_from_home_inferred",
    "continued_from_home_inferred_flag",
    "newly_started_during_encounter_inferred",
    "newly_started_during_encounter_inferred_flag",
    "continuity_inference_rule",
    "medication_episode_id",
    "prescription_segment_count",
    "prescription_segments_json",
    "source_tables_json",
    "source_record_provenance_json",
    "medication_event_build_run_id",
    "medication_event_contract_version",
]

ENCOUNTER_MEDICATION_STATE_COLUMNS = [
    "subject_id",
    "encounter_id",
    "hadm_id",
    "stay_id",
    "review_timestamp",
    "review_timestamp_source",
    "review_time_policy_name",
    "review_timestamp_candidate",
    "review_timestamp_candidate_source",
    "review_time_capped_to_discharge_flag",
    "review_time_validated_flag",
    "discharge_boundary",
    "medication_raw",
    "medication_normalized",
    "medication_standardized",
    "medication_standardized_source",
    "rxnorm_rxcui",
    "rxnorm_matched_term",
    "rxnorm_term_type",
    "ingredient_standardized",
    "ingredient_resolution_status",
    "mapping_confidence",
    "ambiguous_mapping_flag",
    "mapping_candidate_count",
    "medication_mapping_lookup_strategy",
    "medication_class_standardized",
    "medication_status_at_review",
    "active_at_review_flag",
    "continued_from_home_inferred",
    "newly_started_during_encounter_inferred",
    "route",
    "frequency",
    "status",
    "selected_medication_event_id",
    "selected_medication_event_type",
    "active_event_count_at_review",
    "candidate_event_count",
    "source_home_medrecon_flag",
    "source_ed_pyxis_flag",
    "source_hospital_order_flag",
    "source_hospital_admin_flag",
    "source_tables_json",
    "source_record_provenance_json",
    "encounter_medication_state_build_run_id",
    "encounter_medication_state_contract_version",
]

MEDICATION_RXNORM_MAPPING_COLUMNS = [
    "medication_query_key",
    "raw_medication_string",
    "normalized_query_string",
    "lookup_mode",
    "lookup_strategy_used",
    "lookup_status",
    "rxnorm_rxcui",
    "matched_term",
    "matched_term_type",
    "ingredient_rxcui",
    "ingredient_standardized",
    "ingredient_resolution_status",
    "medication_standardized",
    "medication_standardized_source",
    "mapping_confidence",
    "ambiguous_match_flag",
    "candidate_match_count",
    "candidate_rxcuis_json",
    "ambiguity_note",
    "class_assignment_status",
    "class_ids_json",
    "class_labels_json",
    "lookup_timestamp",
    "api_called_flag",
    "mapper_name",
    "mapper_version",
    "rxnorm_version",
    "rxnorm_api_version",
    "mapping_provenance_json",
    "medication_rxnorm_mapping_build_run_id",
    "medication_rxnorm_mapping_contract_version",
]

ENCOUNTER_MEDICATION_SEMANTICS_COLUMNS = [
    "subject_id",
    "encounter_id",
    "hadm_id",
    "stay_id",
    "review_timestamp",
    "review_timestamp_source",
    "medication_standardized",
    "medication_normalized",
    "medication_raw",
    "rxnorm_rxcui",
    "ingredient_standardized",
    "ingredient_resolution_status",
    "mapping_confidence",
    "medication_class_standardized",
    "class_assignment_status",
    "class_system",
    "class_system_version",
    "first_scope_supported_class_flag",
    "medication_status_at_review",
    "active_at_review_flag",
    "continued_from_home_inferred",
    "newly_started_during_encounter_inferred",
    "selected_medication_event_id",
    "selected_medication_event_type",
    "benzodiazepine_heuristic_flag",
    "opioid_heuristic_flag",
    "anticholinergic_heuristic_flag",
    "ppi_heuristic_flag",
    "antipsychotic_heuristic_flag",
    "heuristic_any_supported_class_flag",
    "standardized_vs_heuristic_class_agreement_flag",
    "encounter_medication_semantics_build_run_id",
    "encounter_medication_semantics_contract_version",
]

ENCOUNTER_MEDICATION_BURDEN_COLUMNS = [
    "subject_id",
    "encounter_id",
    "hadm_id",
    "stay_id",
    "review_timestamp",
    "medication_standardized",
    "medication_class_standardized",
    "medication_status_at_review",
    "active_at_review_flag",
    "continued_from_home_inferred",
    "newly_started_during_encounter_inferred",
    "medication_start_context",
    "duration_before_review_hours",
    "duration_before_review_inferable_flag",
    "duration_before_review_lower_bound_flag",
    "scheduled_vs_prn",
    "scheduled_vs_prn_inference_status",
    "dose_value",
    "dose_unit",
    "route",
    "frequency",
    "exact_current_medication_count",
    "current_benzodiazepine_count",
    "current_opioid_count",
    "current_anticholinergic_count",
    "current_ppi_count",
    "current_antipsychotic_count",
    "current_supported_class_count",
    "row_same_class_current_count",
    "same_class_duplicate_therapy_flag",
    "same_class_duplicate_therapy_signal_count",
    "opioid_mme",
    "opioid_mme_status",
    "opioid_mme_required_fields_present_flag",
    "opioid_mme_missing_fields_json",
    "renal_dose_mismatch",
    "renal_dose_mismatch_status",
    "renal_dose_fields_available_flag",
    "renal_context_available_flag",
    "renal_context_age_available_flag",
    "renal_context_sex_available_flag",
    "renal_context_latest_creatinine_time",
    "renal_context_latest_creatinine_value",
    "renal_dose_mismatch_ready_flag",
    "encounter_medication_burden_build_run_id",
    "encounter_medication_burden_contract_version",
]


def validate_encounter_index_artifact(dataframe: pd.DataFrame) -> None:
    """Assert that the encounter-index artifact matches the expected contract."""
    _assert_required_columns(
        dataframe,
        required_columns=ENCOUNTER_INDEX_COLUMNS,
        artifact_name="encounter_index",
    )
    _assert_non_null(
        dataframe,
        columns=["subject_id", "encounter_id", "encounter_source", "encounter_index_build_run_id"],
        artifact_name="encounter_index",
    )
    _assert_unique(
        dataframe,
        key_columns=["encounter_id"],
        artifact_name="encounter_index",
    )
    _assert_allowed_values(
        dataframe,
        column_name="encounter_source",
        allowed_values={"ed_only", "ed_to_inpatient", "hospital_only"},
        artifact_name="encounter_index",
    )
    _assert_json_column(dataframe, "source_tables_json", artifact_name="encounter_index")
    _assert_json_column(
        dataframe,
        "source_record_provenance_json",
        artifact_name="encounter_index",
    )
    _assert_flag_column(dataframe, "linked_ed_stay_flag", artifact_name="encounter_index")
    _assert_flag_column(
        dataframe,
        "linked_hospital_admission_flag",
        artifact_name="encounter_index",
    )
    if (
        dataframe["encounter_index_contract_version"].dropna().astype(str)
        != ENCOUNTER_INDEX_CONTRACT_VERSION
    ).any():
        raise DataLoadError(
            f"Encounter-index artifact must use contract version '{ENCOUNTER_INDEX_CONTRACT_VERSION}'."
        )


def validate_medication_events_artifact(dataframe: pd.DataFrame) -> None:
    """Assert that the medication-events artifact matches the expected contract."""
    _assert_required_columns(
        dataframe,
        required_columns=MEDICATION_EVENT_COLUMNS,
        artifact_name="medication_events",
    )
    _assert_non_null(
        dataframe,
        columns=[
            "subject_id",
            "encounter_id",
            "medication_event_id",
            "medication_event_type",
            "event_source_category",
            "event_source_table",
            "medication_event_build_run_id",
        ],
        artifact_name="medication_events",
    )
    _assert_unique(
        dataframe,
        key_columns=["medication_event_id"],
        artifact_name="medication_events",
    )
    _assert_allowed_values(
        dataframe,
        column_name="medication_event_type",
        allowed_values={"home_medrecon", "ed_pyxis", "hospital_order", "hospital_admin"},
        artifact_name="medication_events",
    )
    _assert_allowed_values(
        dataframe,
        column_name="event_source_category",
        allowed_values={
            "home_medication_reconciliation",
            "ed_medication_event",
            "hospital_medication_order",
            "hospital_administration_event",
        },
        artifact_name="medication_events",
    )
    _assert_json_column(dataframe, "source_tables_json", artifact_name="medication_events")
    _assert_json_column(
        dataframe,
        "source_record_provenance_json",
        artifact_name="medication_events",
    )
    _assert_flag_column(dataframe, "source_home_medrecon", artifact_name="medication_events")
    _assert_flag_column(dataframe, "source_ed_pyxis", artifact_name="medication_events")
    _assert_flag_column(dataframe, "source_hospital_order", artifact_name="medication_events")
    _assert_flag_column(dataframe, "source_hospital_admin", artifact_name="medication_events")
    _assert_flag_column(dataframe, "pharmacy_enriched_flag", artifact_name="medication_events")
    _assert_flag_column(
        dataframe,
        "order_enrichment_applied_flag",
        artifact_name="medication_events",
    )
    _assert_flag_column(
        dataframe,
        "continued_from_home_inferred",
        artifact_name="medication_events",
    )
    _assert_flag_column(
        dataframe,
        "continued_from_home_inferred_flag",
        artifact_name="medication_events",
    )
    _assert_flag_column(
        dataframe,
        "newly_started_during_encounter_inferred",
        artifact_name="medication_events",
    )
    _assert_flag_column(
        dataframe,
        "newly_started_during_encounter_inferred_flag",
        artifact_name="medication_events",
    )
    source_flag_sum = dataframe[
        [
            "source_home_medrecon",
            "source_ed_pyxis",
            "source_hospital_order",
            "source_hospital_admin",
        ]
    ].sum(axis=1)
    if (source_flag_sum != 1).any():
        raise DataLoadError(
            "Medication-events artifact must have exactly one primary source flag set per row."
        )
    if not dataframe["order_enrichment_applied_flag"].equals(dataframe["pharmacy_enriched_flag"]):
        raise DataLoadError(
            "Medication-events artifact must keep order_enrichment_applied_flag aligned with pharmacy_enriched_flag."
        )
    if not dataframe["continued_from_home_inferred_flag"].equals(
        dataframe["continued_from_home_inferred"]
    ):
        raise DataLoadError(
            "Medication-events artifact must keep continued_from_home_inferred_flag aligned with continued_from_home_inferred."
        )
    if not dataframe["newly_started_during_encounter_inferred_flag"].equals(
        dataframe["newly_started_during_encounter_inferred"]
    ):
        raise DataLoadError(
            "Medication-events artifact must keep newly_started_during_encounter_inferred_flag aligned with newly_started_during_encounter_inferred."
        )
    if not dataframe["medication_prestandardized_text"].equals(dataframe["medication_normalized"]):
        raise DataLoadError(
            "Medication-events artifact must carry the current normalized text forward as medication_prestandardized_text."
        )
    if (
        dataframe["medication_event_contract_version"].dropna().astype(str)
        != MEDICATION_EVENTS_CONTRACT_VERSION
    ).any():
        raise DataLoadError(
            f"Medication-events artifact must use contract version '{MEDICATION_EVENTS_CONTRACT_VERSION}'."
        )


def validate_encounter_medication_state_artifact(dataframe: pd.DataFrame) -> None:
    """Assert that the encounter-medication-state artifact matches the expected contract."""
    _assert_required_columns(
        dataframe,
        required_columns=ENCOUNTER_MEDICATION_STATE_COLUMNS,
        artifact_name="encounter_medication_state",
    )
    _assert_non_null(
        dataframe,
        columns=[
            "subject_id",
            "encounter_id",
            "review_timestamp",
            "review_timestamp_source",
            "review_time_policy_name",
            "medication_standardized",
            "medication_standardized_source",
            "ingredient_resolution_status",
            "mapping_confidence",
            "medication_status_at_review",
            "selected_medication_event_id",
            "selected_medication_event_type",
            "encounter_medication_state_build_run_id",
        ],
        artifact_name="encounter_medication_state",
    )
    _assert_unique(
        dataframe,
        key_columns=[
            "subject_id",
            "encounter_id",
            "medication_standardized",
            "review_timestamp",
        ],
        artifact_name="encounter_medication_state",
    )
    _assert_allowed_values(
        dataframe,
        column_name="medication_status_at_review",
        allowed_values={
            "active_at_review_time",
            "inactive_before_review_time",
            "pre_admission_only",
            "activity_uncertain_at_review_time",
        },
        artifact_name="encounter_medication_state",
    )
    _assert_allowed_values(
        dataframe,
        column_name="medication_standardized_source",
        allowed_values={
            "rxnorm_ingredient",
            "rxnorm_term",
            "normalized_text_fallback",
        },
        artifact_name="encounter_medication_state",
    )
    _assert_allowed_values(
        dataframe,
        column_name="ingredient_resolution_status",
        allowed_values={
            "resolved_to_ingredient",
            "resolved_via_related_concept",
            "resolved_term_only_no_ingredient",
            "unresolved_no_match",
            "unresolved_ambiguous_multi_hit",
            "unresolved_cache_only_miss",
            "unresolved_api_error",
        },
        artifact_name="encounter_medication_state",
    )
    _assert_allowed_values(
        dataframe,
        column_name="mapping_confidence",
        allowed_values={"high", "medium", "low", "none"},
        artifact_name="encounter_medication_state",
    )
    _assert_allowed_values(
        dataframe,
        column_name="review_timestamp_source",
        allowed_values={
            "medication_administration",
            "medication_order",
            "lab",
            "vitals",
            "encounter_end",
            "encounter_boundary",
        },
        artifact_name="encounter_medication_state",
    )
    _assert_allowed_values(
        dataframe,
        column_name="review_time_policy_name",
        allowed_values={
            "latest_available",
            "bounded_latest_available",
            "discharge_capped_latest_available",
        },
        artifact_name="encounter_medication_state",
    )
    _assert_json_column(
        dataframe,
        "source_tables_json",
        artifact_name="encounter_medication_state",
    )
    _assert_json_column(
        dataframe,
        "source_record_provenance_json",
        artifact_name="encounter_medication_state",
    )
    _assert_flag_column(
        dataframe,
        "ambiguous_mapping_flag",
        artifact_name="encounter_medication_state",
    )
    _assert_flag_column(
        dataframe,
        "review_time_capped_to_discharge_flag",
        artifact_name="encounter_medication_state",
    )
    _assert_flag_column(
        dataframe,
        "review_time_validated_flag",
        artifact_name="encounter_medication_state",
    )
    _assert_flag_column(
        dataframe,
        "active_at_review_flag",
        artifact_name="encounter_medication_state",
    )
    _assert_flag_column(
        dataframe,
        "continued_from_home_inferred",
        artifact_name="encounter_medication_state",
    )
    _assert_flag_column(
        dataframe,
        "newly_started_during_encounter_inferred",
        artifact_name="encounter_medication_state",
    )
    _assert_flag_column(
        dataframe,
        "source_home_medrecon_flag",
        artifact_name="encounter_medication_state",
    )
    _assert_flag_column(
        dataframe,
        "source_ed_pyxis_flag",
        artifact_name="encounter_medication_state",
    )
    _assert_flag_column(
        dataframe,
        "source_hospital_order_flag",
        artifact_name="encounter_medication_state",
    )
    _assert_flag_column(
        dataframe,
        "source_hospital_admin_flag",
        artifact_name="encounter_medication_state",
    )
    active_flag = pd.to_numeric(dataframe["active_at_review_flag"], errors="coerce").fillna(0).astype(int)
    active_status = dataframe["medication_status_at_review"].astype(str) == "active_at_review_time"
    if not active_flag.eq(active_status.astype(int)).all():
        raise DataLoadError(
            "Encounter-medication-state artifact must keep active_at_review_flag aligned with medication_status_at_review."
        )
    _assert_timestamp_not_after_boundary(
        dataframe,
        timestamp_column="review_timestamp",
        boundary_column="discharge_boundary",
        artifact_name="encounter_medication_state",
    )
    if (
        dataframe["encounter_medication_state_contract_version"].dropna().astype(str)
        != ENCOUNTER_MEDICATION_STATE_CONTRACT_VERSION
    ).any():
        raise DataLoadError(
            "Encounter-medication-state artifact must use contract version "
            f"'{ENCOUNTER_MEDICATION_STATE_CONTRACT_VERSION}'."
        )


def validate_medication_rxnorm_mapping_artifact(dataframe: pd.DataFrame) -> None:
    """Assert that the medication RxNorm mapping artifact matches the expected contract."""
    _assert_required_columns(
        dataframe,
        required_columns=MEDICATION_RXNORM_MAPPING_COLUMNS,
        artifact_name="medication_rxnorm_mapping",
    )
    _assert_non_null(
        dataframe,
        columns=[
            "medication_query_key",
            "normalized_query_string",
            "lookup_mode",
            "lookup_strategy_used",
            "lookup_status",
            "ingredient_resolution_status",
            "medication_standardized",
            "medication_standardized_source",
            "mapping_confidence",
            "lookup_timestamp",
            "mapper_name",
            "mapper_version",
            "rxnorm_api_version",
            "medication_rxnorm_mapping_build_run_id",
        ],
        artifact_name="medication_rxnorm_mapping",
    )
    _assert_unique(
        dataframe,
        key_columns=["medication_query_key"],
        artifact_name="medication_rxnorm_mapping",
    )
    _assert_allowed_values(
        dataframe,
        column_name="lookup_mode",
        allowed_values={"cache_first", "cache_only"},
        artifact_name="medication_rxnorm_mapping",
    )
    _assert_allowed_values(
        dataframe,
        column_name="lookup_status",
        allowed_values={
            "resolved",
            "resolved_term_only_no_ingredient",
            "unresolved_no_match",
            "unresolved_ambiguous_multi_hit",
            "unresolved_cache_only_miss",
            "unresolved_api_error",
        },
        artifact_name="medication_rxnorm_mapping",
    )
    _assert_allowed_values(
        dataframe,
        column_name="ingredient_resolution_status",
        allowed_values={
            "resolved_to_ingredient",
            "resolved_via_related_concept",
            "resolved_term_only_no_ingredient",
            "unresolved_no_match",
            "unresolved_ambiguous_multi_hit",
            "unresolved_cache_only_miss",
            "unresolved_api_error",
        },
        artifact_name="medication_rxnorm_mapping",
    )
    _assert_allowed_values(
        dataframe,
        column_name="medication_standardized_source",
        allowed_values={
            "rxnorm_ingredient",
            "rxnorm_term",
            "normalized_text_fallback",
        },
        artifact_name="medication_rxnorm_mapping",
    )
    _assert_allowed_values(
        dataframe,
        column_name="mapping_confidence",
        allowed_values={"high", "medium", "low", "none"},
        artifact_name="medication_rxnorm_mapping",
    )
    _assert_json_column(
        dataframe,
        "candidate_rxcuis_json",
        artifact_name="medication_rxnorm_mapping",
    )
    _assert_json_column(
        dataframe,
        "class_ids_json",
        artifact_name="medication_rxnorm_mapping",
    )
    _assert_json_column(
        dataframe,
        "class_labels_json",
        artifact_name="medication_rxnorm_mapping",
    )
    _assert_json_column(
        dataframe,
        "mapping_provenance_json",
        artifact_name="medication_rxnorm_mapping",
    )
    _assert_flag_column(
        dataframe,
        "ambiguous_match_flag",
        artifact_name="medication_rxnorm_mapping",
    )
    _assert_flag_column(
        dataframe,
        "api_called_flag",
        artifact_name="medication_rxnorm_mapping",
    )
    if (
        dataframe["medication_rxnorm_mapping_contract_version"].dropna().astype(str)
        != MEDICATION_RXNORM_MAPPING_CONTRACT_VERSION
    ).any():
        raise DataLoadError(
            "Medication RxNorm mapping artifact must use contract version "
            f"'{MEDICATION_RXNORM_MAPPING_CONTRACT_VERSION}'."
        )


def validate_encounter_medication_semantics_artifact(dataframe: pd.DataFrame) -> None:
    """Assert that the encounter-medication-semantics artifact matches the expected contract."""
    _assert_required_columns(
        dataframe,
        required_columns=ENCOUNTER_MEDICATION_SEMANTICS_COLUMNS,
        artifact_name="encounter_medication_semantics",
    )
    _assert_non_null(
        dataframe,
        columns=[
            "subject_id",
            "encounter_id",
            "review_timestamp",
            "medication_standardized",
            "ingredient_resolution_status",
            "mapping_confidence",
            "medication_class_standardized",
            "class_assignment_status",
            "class_system",
            "class_system_version",
            "first_scope_supported_class_flag",
            "active_at_review_flag",
            "encounter_medication_semantics_build_run_id",
        ],
        artifact_name="encounter_medication_semantics",
    )
    _assert_unique(
        dataframe,
        key_columns=[
            "subject_id",
            "encounter_id",
            "medication_standardized",
            "review_timestamp",
        ],
        artifact_name="encounter_medication_semantics",
    )
    _assert_allowed_values(
        dataframe,
        column_name="medication_class_standardized",
        allowed_values=set(FIRST_SCOPE_SUPPORTED_CLASSES) | {"unresolved"},
        artifact_name="encounter_medication_semantics",
    )
    _assert_allowed_values(
        dataframe,
        column_name="class_assignment_status",
        allowed_values={
            "supported_scope_class_assigned",
            "supported_scope_unresolved",
            "no_standardized_ingredient_available",
            "class_enrichment_not_attempted",
        },
        artifact_name="encounter_medication_semantics",
    )
    _assert_allowed_values(
        dataframe,
        column_name="medication_status_at_review",
        allowed_values={
            "active_at_review_time",
            "inactive_before_review_time",
            "pre_admission_only",
            "activity_uncertain_at_review_time",
        },
        artifact_name="encounter_medication_semantics",
    )
    for column_name in [
        "first_scope_supported_class_flag",
        "active_at_review_flag",
        "continued_from_home_inferred",
        "newly_started_during_encounter_inferred",
        "benzodiazepine_heuristic_flag",
        "opioid_heuristic_flag",
        "anticholinergic_heuristic_flag",
        "ppi_heuristic_flag",
        "antipsychotic_heuristic_flag",
        "heuristic_any_supported_class_flag",
        "standardized_vs_heuristic_class_agreement_flag",
    ]:
        _assert_flag_column(
            dataframe,
            column_name,
            artifact_name="encounter_medication_semantics",
        )
    if (
        dataframe["encounter_medication_semantics_contract_version"].dropna().astype(str)
        != ENCOUNTER_MEDICATION_SEMANTICS_CONTRACT_VERSION
    ).any():
        raise DataLoadError(
            "Encounter-medication-semantics artifact must use contract version "
            f"'{ENCOUNTER_MEDICATION_SEMANTICS_CONTRACT_VERSION}'."
        )


def validate_encounter_medication_burden_artifact(dataframe: pd.DataFrame) -> None:
    """Assert that the encounter-medication-burden artifact matches the expected contract."""
    _assert_required_columns(
        dataframe,
        required_columns=ENCOUNTER_MEDICATION_BURDEN_COLUMNS,
        artifact_name="encounter_medication_burden",
    )
    _assert_non_null(
        dataframe,
        columns=[
            "subject_id",
            "encounter_id",
            "review_timestamp",
            "medication_standardized",
            "medication_class_standardized",
            "active_at_review_flag",
            "scheduled_vs_prn",
            "scheduled_vs_prn_inference_status",
            "exact_current_medication_count",
            "same_class_duplicate_therapy_flag",
            "same_class_duplicate_therapy_signal_count",
            "opioid_mme_status",
            "opioid_mme_required_fields_present_flag",
            "opioid_mme_missing_fields_json",
            "renal_dose_mismatch_status",
            "renal_dose_fields_available_flag",
            "renal_context_available_flag",
            "renal_context_age_available_flag",
            "renal_context_sex_available_flag",
            "renal_dose_mismatch_ready_flag",
            "encounter_medication_burden_build_run_id",
        ],
        artifact_name="encounter_medication_burden",
    )
    _assert_unique(
        dataframe,
        key_columns=[
            "subject_id",
            "encounter_id",
            "medication_standardized",
            "review_timestamp",
        ],
        artifact_name="encounter_medication_burden",
    )
    _assert_allowed_values(
        dataframe,
        column_name="medication_class_standardized",
        allowed_values=set(FIRST_SCOPE_SUPPORTED_CLASSES) | {"unresolved"},
        artifact_name="encounter_medication_burden",
    )
    _assert_allowed_values(
        dataframe,
        column_name="medication_status_at_review",
        allowed_values={
            "active_at_review_time",
            "inactive_before_review_time",
            "pre_admission_only",
            "activity_uncertain_at_review_time",
        },
        artifact_name="encounter_medication_burden",
    )
    _assert_allowed_values(
        dataframe,
        column_name="scheduled_vs_prn",
        allowed_values={"scheduled", "prn", "unknown"},
        artifact_name="encounter_medication_burden",
    )
    _assert_allowed_values(
        dataframe,
        column_name="scheduled_vs_prn_inference_status",
        allowed_values={"explicit_prn", "explicit_scheduled", "unknown"},
        artifact_name="encounter_medication_burden",
    )
    _assert_allowed_values(
        dataframe,
        column_name="medication_start_context",
        allowed_values={
            "continued_from_home",
            "new_start_during_encounter",
            "mixed_evidence",
            "neither_or_unknown",
        },
        artifact_name="encounter_medication_burden",
    )
    _assert_allowed_values(
        dataframe,
        column_name="opioid_mme_status",
        allowed_values={
            "not_opioid",
            "missing_required_fields",
            "ready_for_future_conversion",
        },
        artifact_name="encounter_medication_burden",
    )
    _assert_allowed_values(
        dataframe,
        column_name="renal_dose_mismatch_status",
        allowed_values={
            "not_current_medication",
            "insufficient_inputs",
            "ready_for_future_logic",
        },
        artifact_name="encounter_medication_burden",
    )
    _assert_json_column(
        dataframe,
        "opioid_mme_missing_fields_json",
        artifact_name="encounter_medication_burden",
    )
    for column_name in [
        "active_at_review_flag",
        "continued_from_home_inferred",
        "newly_started_during_encounter_inferred",
        "duration_before_review_inferable_flag",
        "duration_before_review_lower_bound_flag",
        "same_class_duplicate_therapy_flag",
        "opioid_mme_required_fields_present_flag",
        "renal_dose_fields_available_flag",
        "renal_context_available_flag",
        "renal_context_age_available_flag",
        "renal_context_sex_available_flag",
        "renal_dose_mismatch_ready_flag",
    ]:
        _assert_flag_column(
            dataframe,
            column_name,
            artifact_name="encounter_medication_burden",
        )
    if (
        dataframe["encounter_medication_burden_contract_version"].dropna().astype(str)
        != ENCOUNTER_MEDICATION_BURDEN_CONTRACT_VERSION
    ).any():
        raise DataLoadError(
            "Encounter-medication-burden artifact must use contract version "
            f"'{ENCOUNTER_MEDICATION_BURDEN_CONTRACT_VERSION}'."
        )


def _assert_required_columns(
    dataframe: pd.DataFrame,
    *,
    required_columns: list[str],
    artifact_name: str,
) -> None:
    missing_columns = sorted(set(required_columns) - set(dataframe.columns))
    if missing_columns:
        raise DataLoadError(
            f"Artifact '{artifact_name}' is missing required columns: {', '.join(missing_columns)}"
        )


def _assert_non_null(
    dataframe: pd.DataFrame,
    *,
    columns: list[str],
    artifact_name: str,
) -> None:
    for column_name in columns:
        if dataframe[column_name].isna().any():
            raise DataLoadError(
                f"Artifact '{artifact_name}' has null values in required column '{column_name}'."
            )


def _assert_unique(
    dataframe: pd.DataFrame,
    *,
    key_columns: list[str],
    artifact_name: str,
) -> None:
    duplicate_mask = dataframe.duplicated(subset=key_columns, keep=False)
    if duplicate_mask.any():
        raise DataLoadError(
            f"Artifact '{artifact_name}' has duplicate keys for columns: {', '.join(key_columns)}"
        )


def _assert_allowed_values(
    dataframe: pd.DataFrame,
    *,
    column_name: str,
    allowed_values: set[str],
    artifact_name: str,
) -> None:
    unexpected_values = sorted(
        {
            str(value)
            for value in dataframe[column_name].dropna().astype(str).tolist()
            if str(value) not in allowed_values
        }
    )
    if unexpected_values:
        raise DataLoadError(
            f"Artifact '{artifact_name}' has unsupported values in '{column_name}': {', '.join(unexpected_values)}"
        )


def _assert_json_column(
    dataframe: pd.DataFrame,
    column_name: str,
    *,
    artifact_name: str,
) -> None:
    invalid_examples: list[str] = []
    for value in dataframe[column_name].dropna().tolist():
        try:
            json.loads(str(value))
        except json.JSONDecodeError:
            invalid_examples.append(str(value))
            if len(invalid_examples) >= 3:
                break
    if invalid_examples:
        raise DataLoadError(
            f"Artifact '{artifact_name}' has invalid JSON values in '{column_name}': {', '.join(invalid_examples)}"
        )


def _assert_flag_column(
    dataframe: pd.DataFrame,
    column_name: str,
    *,
    artifact_name: str,
) -> None:
    values = set(pd.to_numeric(dataframe[column_name], errors="coerce").dropna().astype(int).tolist())
    if not values.issubset({0, 1}):
        raise DataLoadError(
            f"Artifact '{artifact_name}' has non-binary values in flag column '{column_name}'."
        )


def _assert_timestamp_not_after_boundary(
    dataframe: pd.DataFrame,
    *,
    timestamp_column: str,
    boundary_column: str,
    artifact_name: str,
) -> None:
    if dataframe.empty:
        return
    timestamp_values = pd.to_datetime(dataframe[timestamp_column], errors="coerce")
    boundary_values = pd.to_datetime(dataframe[boundary_column], errors="coerce")
    violation_mask = boundary_values.notna() & timestamp_values.notna() & (timestamp_values > boundary_values)
    if violation_mask.any():
        raise DataLoadError(
            f"Artifact '{artifact_name}' has {int(violation_mask.sum()):,} rows where "
            f"'{timestamp_column}' exceeds '{boundary_column}'."
        )
