"""Synthetic tests for Labels v1 on the 65+ first-scope subset."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from opti_med.data_access.artifact_schemas import (
    ENCOUNTER_INDEX_COLUMNS,
    ENCOUNTER_INDEX_CONTRACT_VERSION,
    ENCOUNTER_MEDICATION_BURDEN_CONTRACT_VERSION,
    ENCOUNTER_MEDICATION_FIRST_SCOPE_COLUMNS,
    ENCOUNTER_MEDICATION_FIRST_SCOPE_CONTRACT_VERSION,
    ENCOUNTER_MEDICATION_LABELS_V1_FIRST_SCOPE_CONTRACT_VERSION,
    ENCOUNTER_MEDICATION_SEMANTICS_CONTRACT_VERSION,
    ENCOUNTER_MEDICATION_STATE_COLUMNS,
    ENCOUNTER_MEDICATION_STATE_CONTRACT_VERSION,
    MEDICATION_EVENT_COLUMNS,
    MEDICATION_EVENTS_CONTRACT_VERSION,
    MEDICATION_RXNORM_MAPPING_COLUMNS,
    MEDICATION_RXNORM_MAPPING_CONTRACT_VERSION,
    validate_encounter_medication_labels_v1_first_scope_artifact,
)
from opti_med.labels.first_scope import (
    build_first_scope_label_qc_report,
    build_first_scope_labels,
)


class FirstScopeLabelsV1Tests(unittest.TestCase):
    def test_clear_stop_after_review_is_positive_primary_label(self) -> None:
        first_scope = make_first_scope([{}])
        labels = build_first_scope_labels(
            encounter_medication_first_scope=first_scope,
            encounter_medication_state=make_state(first_scope),
            medication_events=make_medication_events(
                [
                    {
                        "medication_event_id": "order-review",
                        "medication_event_type": "hospital_order",
                        "raw_medication_name": "Oxycodone 10 mg",
                        "medication_name": "Oxycodone 10 mg",
                        "medication_normalized": "oxycodone",
                        "event_time": "2125-03-20 08:00:00",
                        "starttime": "2125-03-20 08:00:00",
                        "stoptime": "2125-03-20 14:00:00",
                        "route": "PO",
                        "frequency": "BID",
                        "status": "active",
                        "dose_value": "10",
                        "dose_unit": "mg",
                    }
                ]
            ),
            medication_rxnorm_mapping=make_mapping(first_scope),
            encounter_index=make_encounter_index(),
        )

        validate_encounter_medication_labels_v1_first_scope_artifact(labels)
        row = labels.iloc[0]
        self.assertEqual(
            row["primary_action_label"],
            "stopped_or_deintensified_before_discharge",
        )
        self.assertEqual(row["primary_action_label_reason"], "clear_post_review_stop_before_discharge")
        self.assertEqual(int(row["unknown_or_insufficient_evidence_flag"]), 0)
        self.assertEqual(int(row["post_review_stop_evidence_flag"]), 1)

    def test_clear_continuation_is_negative_primary_label(self) -> None:
        first_scope = make_first_scope([{}])
        labels = build_first_scope_labels(
            encounter_medication_first_scope=first_scope,
            encounter_medication_state=make_state(first_scope),
            medication_events=make_medication_events(
                [
                    {
                        "medication_event_id": "order-review",
                        "medication_event_type": "hospital_order",
                        "raw_medication_name": "Oxycodone 10 mg",
                        "medication_name": "Oxycodone 10 mg",
                        "medication_normalized": "oxycodone",
                        "event_time": "2125-03-20 08:00:00",
                        "starttime": "2125-03-20 08:00:00",
                        "stoptime": "2125-03-20 20:00:00",
                        "route": "PO",
                        "frequency": "BID",
                        "status": "active",
                        "dose_value": "10",
                        "dose_unit": "mg",
                    }
                ]
            ),
            medication_rxnorm_mapping=make_mapping(first_scope),
            encounter_index=make_encounter_index(),
        )

        row = labels.iloc[0]
        self.assertEqual(
            row["primary_action_label"],
            "no_clear_stop_or_deintensification_before_discharge",
        )
        self.assertEqual(int(row["unknown_or_insufficient_evidence_flag"]), 0)
        self.assertEqual(int(row["post_review_continuation_evidence_flag"]), 1)

    def test_ambiguous_dose_change_stays_explicitly_unknown(self) -> None:
        first_scope = make_first_scope([{}])
        labels = build_first_scope_labels(
            encounter_medication_first_scope=first_scope,
            encounter_medication_state=make_state(first_scope),
            medication_events=make_medication_events(
                [
                    {
                        "medication_event_id": "order-review",
                        "medication_event_type": "hospital_order",
                        "raw_medication_name": "Oxycodone 10 mg",
                        "medication_name": "Oxycodone 10 mg",
                        "medication_normalized": "oxycodone",
                        "event_time": "2125-03-20 08:00:00",
                        "starttime": "2125-03-20 08:00:00",
                        "stoptime": "2125-03-20 12:00:00",
                        "route": "PO",
                        "frequency": "BID",
                        "status": "active",
                        "dose_value": "10",
                        "dose_unit": "mg",
                    },
                    {
                        "medication_event_id": "order-lower-prn",
                        "medication_event_type": "hospital_order",
                        "raw_medication_name": "Oxycodone 5 mg",
                        "medication_name": "Oxycodone 5 mg",
                        "medication_normalized": "oxycodone",
                        "event_time": "2125-03-20 12:00:00",
                        "starttime": "2125-03-20 12:00:00",
                        "stoptime": "2125-03-20 18:00:00",
                        "route": "PO",
                        "frequency": "Q6H PRN",
                        "status": "active",
                        "dose_value": "5",
                        "dose_unit": "mg",
                    },
                ]
            ),
            medication_rxnorm_mapping=make_mapping(first_scope),
            encounter_index=make_encounter_index(),
        )

        row = labels.iloc[0]
        self.assertEqual(row["primary_action_label"], "action_undetermined")
        self.assertEqual(int(row["unknown_or_insufficient_evidence_flag"]), 1)
        self.assertIn("ambiguous_post_review_medication_change", row["primary_action_label_reason"])

    def test_excluded_acute_medication_sets_hard_exclusion_flag(self) -> None:
        first_scope = make_first_scope(
            [
                {
                    "medication_standardized": "fentanyl",
                    "medication_normalized": "fentanyl",
                    "medication_raw": "Fentanyl",
                    "ingredient_standardized": "fentanyl",
                    "rxnorm_rxcui": "4337",
                    "rxnorm_matched_term": "fentanyl",
                    "selected_medication_event_id": "admin-review",
                    "selected_medication_event_type": "hospital_admin",
                    "dose_value": "25",
                    "dose_unit": "mcg",
                    "route": "IV",
                    "frequency": "ONCE",
                    "duration_before_review_hours": 1.0,
                }
            ]
        )
        labels = build_first_scope_labels(
            encounter_medication_first_scope=first_scope,
            encounter_medication_state=make_state(first_scope),
            medication_events=make_medication_events(
                [
                    {
                        "medication_event_id": "admin-review",
                        "medication_event_type": "hospital_admin",
                        "raw_medication_name": "Fentanyl 25 mcg",
                        "medication_name": "Fentanyl 25 mcg",
                        "medication_normalized": "fentanyl",
                        "event_time": "2125-03-20 10:00:00",
                        "starttime": "2125-03-20 10:00:00",
                        "stoptime": None,
                        "route": "IV",
                        "frequency": None,
                        "status": "given",
                        "dose_value": "25",
                        "dose_unit": "mcg",
                    }
                ]
            ),
            medication_rxnorm_mapping=make_mapping(first_scope),
            encounter_index=make_encounter_index(),
        )

        row = labels.iloc[0]
        self.assertEqual(int(row["acute_life_sustaining_medication"]), 1)
        self.assertEqual(int(row["excluded_from_primary_training"]), 1)

    def test_qc_report_mentions_unknown_and_exclusion_rates(self) -> None:
        first_scope = make_first_scope([{}, {"medication_standardized": "fentanyl", "medication_normalized": "fentanyl", "medication_raw": "Fentanyl", "ingredient_standardized": "fentanyl", "rxnorm_rxcui": "4337", "rxnorm_matched_term": "fentanyl", "selected_medication_event_id": "admin-review", "selected_medication_event_type": "hospital_admin", "dose_value": "25", "dose_unit": "mcg", "route": "IV", "frequency": "ONCE", "duration_before_review_hours": 1.0}])
        labels = build_first_scope_labels(
            encounter_medication_first_scope=first_scope,
            encounter_medication_state=make_state(first_scope),
            medication_events=make_medication_events(
                [
                    {
                        "medication_event_id": "order-review",
                        "medication_event_type": "hospital_order",
                        "raw_medication_name": "Oxycodone 10 mg",
                        "medication_name": "Oxycodone 10 mg",
                        "medication_normalized": "oxycodone",
                        "event_time": "2125-03-20 08:00:00",
                        "starttime": "2125-03-20 08:00:00",
                        "stoptime": "2125-03-20 14:00:00",
                        "route": "PO",
                        "frequency": "BID",
                        "status": "active",
                        "dose_value": "10",
                        "dose_unit": "mg",
                    },
                    {
                        "medication_event_id": "admin-review",
                        "medication_event_type": "hospital_admin",
                        "raw_medication_name": "Fentanyl 25 mcg",
                        "medication_name": "Fentanyl 25 mcg",
                        "medication_normalized": "fentanyl",
                        "event_time": "2125-03-20 10:00:00",
                        "starttime": "2125-03-20 10:00:00",
                        "stoptime": None,
                        "route": "IV",
                        "frequency": None,
                        "status": "given",
                        "dose_value": "25",
                        "dose_unit": "mcg",
                    },
                ]
            ),
            medication_rxnorm_mapping=make_mapping(first_scope),
            encounter_index=make_encounter_index(),
        )

        report = build_first_scope_label_qc_report(encounter_medication_labels=labels)
        self.assertIn("Unknown And Exclusion Rates", report)
        self.assertIn("Class-Wise Primary Label Distribution", report)


def make_first_scope(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a minimally valid synthetic first-scope artifact."""
    records: list[dict[str, object]] = []
    for index, overrides in enumerate(rows):
        row = {column_name: None for column_name in ENCOUNTER_MEDICATION_FIRST_SCOPE_COLUMNS}
        medication_standardized = str(overrides.get("medication_standardized", "oxycodone"))
        medication_normalized = str(overrides.get("medication_normalized", medication_standardized))
        medication_raw = str(overrides.get("medication_raw", medication_standardized.title()))
        ingredient_standardized = str(
            overrides.get("ingredient_standardized", medication_standardized)
        )
        rxnorm_rxcui = str(overrides.get("rxnorm_rxcui", "7804"))
        rxnorm_matched_term = str(overrides.get("rxnorm_matched_term", medication_standardized))
        selected_event_id = str(overrides.get("selected_medication_event_id", "order-review"))
        selected_event_type = str(
            overrides.get("selected_medication_event_type", "hospital_order")
        )
        dose_unit = str(overrides.get("dose_unit", "mg"))
        dose_value = str(overrides.get("dose_value", "10"))
        route = str(overrides.get("route", "PO"))
        frequency = overrides.get("frequency", "BID")
        frequency = None if frequency is None else str(frequency)
        row.update(
            {
                "subject_id": 101,
                "encounter_id": "hadm:1",
                "hadm_id": 1,
                "stay_id": 11,
                "review_timestamp": "2125-03-20 10:00:00",
                "review_timestamp_source": "medication_order",
                "age_proxy": 75,
                "age_group": "75-84",
                "medication_standardized": medication_standardized,
                "medication_standardized_source": "rxnorm_ingredient",
                "medication_normalized": medication_normalized,
                "medication_raw": medication_raw,
                "rxnorm_rxcui": rxnorm_rxcui,
                "rxnorm_matched_term": rxnorm_matched_term,
                "rxnorm_term_type": "IN",
                "ingredient_standardized": ingredient_standardized,
                "ingredient_resolution_status": "resolved_to_ingredient",
                "mapping_confidence": "high",
                "ambiguous_mapping_flag": 0,
                "mapping_candidate_count": 1,
                "medication_mapping_lookup_strategy": "exact_or_normalized_raw_query",
                "medication_class_standardized": "opioid",
                "class_assignment_status": "supported_scope_class_assigned",
                "class_system": "opti_med_first_scope",
                "class_system_version": "2026.03_first_pass",
                "first_scope_supported_class_flag": 1,
                "medication_status_at_review": "active_at_review_time",
                "active_at_review_flag": 1,
                "continued_from_home_inferred": 0,
                "newly_started_during_encounter_inferred": 1,
                "selected_medication_event_id": selected_event_id,
                "selected_medication_event_type": selected_event_type,
                "benzodiazepine_heuristic_flag": 0,
                "opioid_heuristic_flag": 1,
                "anticholinergic_heuristic_flag": 0,
                "ppi_heuristic_flag": 0,
                "antipsychotic_heuristic_flag": 0,
                "heuristic_any_supported_class_flag": 1,
                "standardized_vs_heuristic_class_agreement_flag": 1,
                "encounter_medication_semantics_build_run_id": "semantics-run",
                "encounter_medication_semantics_contract_version": (
                    ENCOUNTER_MEDICATION_SEMANTICS_CONTRACT_VERSION
                ),
                "medication_start_context": "new_start_during_encounter",
                "duration_before_review_hours": 2.0,
                "duration_before_review_inferable_flag": 1,
                "duration_before_review_lower_bound_flag": 0,
                "scheduled_vs_prn": "scheduled" if frequency != "Q6H PRN" else "prn",
                "scheduled_vs_prn_inference_status": (
                    "explicit_scheduled" if frequency != "Q6H PRN" else "explicit_prn"
                ),
                "dose_value": dose_value,
                "dose_unit": dose_unit,
                "route": route,
                "frequency": frequency,
                "exact_current_medication_count": 1,
                "current_benzodiazepine_count": 0,
                "current_opioid_count": 1,
                "current_anticholinergic_count": 0,
                "current_ppi_count": 0,
                "current_antipsychotic_count": 0,
                "current_supported_class_count": 1,
                "row_same_class_current_count": 1,
                "same_class_duplicate_therapy_flag": 0,
                "same_class_duplicate_therapy_signal_count": 0,
                "opioid_mme": None,
                "opioid_mme_status": "missing_required_fields",
                "opioid_mme_required_fields_present_flag": 0,
                "opioid_mme_missing_fields_json": json.dumps(["conversion_table"]),
                "renal_dose_mismatch": None,
                "renal_dose_mismatch_status": "insufficient_inputs",
                "renal_dose_fields_available_flag": 0,
                "renal_context_available_flag": 0,
                "renal_context_age_available_flag": 1,
                "renal_context_sex_available_flag": 1,
                "renal_context_latest_creatinine_time": "2125-03-20 09:00:00",
                "renal_context_latest_creatinine_value": 1.2,
                "renal_dose_mismatch_ready_flag": 0,
                "encounter_medication_burden_build_run_id": "burden-run",
                "encounter_medication_burden_contract_version": (
                    ENCOUNTER_MEDICATION_BURDEN_CONTRACT_VERSION
                ),
                "encounter_medication_first_scope_build_run_id": "first-scope-run",
                "encounter_medication_first_scope_contract_version": (
                    ENCOUNTER_MEDICATION_FIRST_SCOPE_CONTRACT_VERSION
                ),
            }
        )
        row.update(overrides)
        records.append(row)
    return pd.DataFrame(records)


def make_state(first_scope: pd.DataFrame) -> pd.DataFrame:
    """Build a minimally valid RxNorm-enriched state artifact aligned to first-scope."""
    records: list[dict[str, object]] = []
    for row in first_scope.to_dict(orient="records"):
        state_row = {column_name: None for column_name in ENCOUNTER_MEDICATION_STATE_COLUMNS}
        selected_event_type = str(row["selected_medication_event_type"])
        state_row.update(
            {
                "subject_id": row["subject_id"],
                "encounter_id": row["encounter_id"],
                "hadm_id": row["hadm_id"],
                "stay_id": row["stay_id"],
                "review_timestamp": row["review_timestamp"],
                "review_timestamp_source": row["review_timestamp_source"],
                "age_proxy": row["age_proxy"],
                "age_group": row["age_group"],
                "review_time_policy_name": "discharge_capped_latest_available",
                "review_timestamp_candidate": row["review_timestamp"],
                "review_timestamp_candidate_source": row["review_timestamp_source"],
                "review_time_capped_to_discharge_flag": 0,
                "review_time_validated_flag": 1,
                "discharge_boundary": "2125-03-20 18:00:00",
                "medication_raw": row["medication_raw"],
                "medication_normalized": row["medication_normalized"],
                "medication_standardized": row["medication_standardized"],
                "medication_standardized_source": row["medication_standardized_source"],
                "rxnorm_rxcui": row["rxnorm_rxcui"],
                "rxnorm_matched_term": row["rxnorm_matched_term"],
                "rxnorm_term_type": row["rxnorm_term_type"],
                "ingredient_standardized": row["ingredient_standardized"],
                "ingredient_resolution_status": row["ingredient_resolution_status"],
                "mapping_confidence": row["mapping_confidence"],
                "ambiguous_mapping_flag": row["ambiguous_mapping_flag"],
                "mapping_candidate_count": row["mapping_candidate_count"],
                "medication_mapping_lookup_strategy": row["medication_mapping_lookup_strategy"],
                "medication_class_standardized": row["medication_class_standardized"],
                "medication_status_at_review": row["medication_status_at_review"],
                "active_at_review_flag": row["active_at_review_flag"],
                "continued_from_home_inferred": row["continued_from_home_inferred"],
                "newly_started_during_encounter_inferred": row["newly_started_during_encounter_inferred"],
                "route": row["route"],
                "frequency": row["frequency"],
                "status": "active",
                "selected_medication_event_id": row["selected_medication_event_id"],
                "selected_medication_event_type": selected_event_type,
                "active_event_count_at_review": 1,
                "candidate_event_count": 1,
                "source_home_medrecon_flag": int(selected_event_type == "home_medrecon"),
                "source_ed_pyxis_flag": int(selected_event_type == "ed_pyxis"),
                "source_hospital_order_flag": int(selected_event_type == "hospital_order"),
                "source_hospital_admin_flag": int(selected_event_type == "hospital_admin"),
                "source_tables_json": json.dumps(
                    ["prescriptions"] if selected_event_type == "hospital_order" else ["emar"]
                ),
                "source_record_provenance_json": json.dumps({"synthetic": True}),
                "encounter_medication_state_build_run_id": "state-run",
                "encounter_medication_state_contract_version": (
                    ENCOUNTER_MEDICATION_STATE_CONTRACT_VERSION
                ),
            }
        )
        records.append(state_row)
    return pd.DataFrame(records, columns=ENCOUNTER_MEDICATION_STATE_COLUMNS)


def make_mapping(first_scope: pd.DataFrame) -> pd.DataFrame:
    """Build a minimally valid RxNorm mapping artifact for the test medications."""
    records: list[dict[str, object]] = []
    seen_query_keys: set[str] = set()
    for row in first_scope.to_dict(orient="records"):
        query_key = str(row["medication_normalized"])
        if query_key in seen_query_keys:
            continue
        seen_query_keys.add(query_key)
        mapping_row = {column_name: None for column_name in MEDICATION_RXNORM_MAPPING_COLUMNS}
        mapping_row.update(
            {
                "medication_query_key": query_key,
                "medication_raw": row["medication_raw"],
                "medication_normalized": row["medication_normalized"],
                "raw_medication_string": row["medication_raw"],
                "normalized_query_string": query_key,
                "lookup_mode": "cache_only",
                "lookup_strategy_used": "exact_or_normalized_raw_query",
                "lookup_status": "resolved",
                "rxnorm_rxcui": row["rxnorm_rxcui"],
                "matched_term": row["rxnorm_matched_term"],
                "matched_term_type": row["rxnorm_term_type"],
                "ingredient_rxcui": row["rxnorm_rxcui"],
                "ingredient_standardized": row["ingredient_standardized"],
                "ingredient_resolution_status": row["ingredient_resolution_status"],
                "medication_standardized": row["medication_standardized"],
                "medication_standardized_source": row["medication_standardized_source"],
                "mapping_confidence": row["mapping_confidence"],
                "ambiguous_match_flag": row["ambiguous_mapping_flag"],
                "candidate_match_count": row["mapping_candidate_count"],
                "candidate_rxcuis_json": json.dumps([row["rxnorm_rxcui"]]),
                "ambiguity_note": None,
                "unresolved_reason": None,
                "class_assignment_status": "supported_scope_class_assigned",
                "class_ids_json": json.dumps(
                    [f"opti_med_first_scope:{row['medication_class_standardized']}"]
                ),
                "class_labels_json": json.dumps([row["medication_class_standardized"]]),
                "lookup_timestamp": "2125-03-20 00:00:00",
                "api_called_flag": 0,
                "mapper_name": "synthetic_test_mapper",
                "mapper_version": "1.0",
                "rxnorm_version": "test",
                "rxnorm_api_version": "test",
                "mapping_provenance_json": json.dumps({"synthetic": True}),
                "medication_rxnorm_mapping_build_run_id": "mapping-run",
                "medication_rxnorm_mapping_contract_version": (
                    MEDICATION_RXNORM_MAPPING_CONTRACT_VERSION
                ),
            }
        )
        records.append(mapping_row)
    return pd.DataFrame(records, columns=MEDICATION_RXNORM_MAPPING_COLUMNS)


def make_medication_events(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a minimally valid medication-events artifact."""
    records: list[dict[str, object]] = []
    for overrides in rows:
        event_type = str(overrides["medication_event_type"])
        source_category, source_table = {
            "hospital_order": ("hospital_medication_order", "prescriptions"),
            "hospital_admin": ("hospital_administration_event", "emar"),
            "ed_pyxis": ("ed_medication_event", "pyxis"),
            "home_medrecon": ("home_medication_reconciliation", "medrecon"),
        }[event_type]
        row = {column_name: None for column_name in MEDICATION_EVENT_COLUMNS}
        row.update(
            {
                "subject_id": 101,
                "hadm_id": 1,
                "stay_id": 11,
                "encounter_id": "hadm:1",
                "encounter_source": "hospital_only",
                "encounter_start": "2125-03-20 07:00:00",
                "encounter_end": "2125-03-20 18:00:00",
                "event_source_category": source_category,
                "event_source_table": source_table,
                "medication_prestandardized_text": overrides["medication_normalized"],
                "pharmacy_id": None,
                "poe_id": None,
                "emar_id": None,
                "emar_seq": None,
                "source_home_medrecon": int(event_type == "home_medrecon"),
                "source_ed_pyxis": int(event_type == "ed_pyxis"),
                "source_hospital_order": int(event_type == "hospital_order"),
                "source_hospital_admin": int(event_type == "hospital_admin"),
                "pharmacy_enriched_flag": 0,
                "order_enrichment_applied_flag": 0,
                "order_enrichment_source_table": None,
                "continued_from_home_inferred": 0,
                "continued_from_home_inferred_flag": 0,
                "newly_started_during_encounter_inferred": 1,
                "newly_started_during_encounter_inferred_flag": 1,
                "continuity_inference_rule": None,
                "medication_episode_id": None,
                "prescription_segment_count": 1,
                "prescription_segments_json": json.dumps([]),
                "source_tables_json": json.dumps([source_table]),
                "source_record_provenance_json": json.dumps({"synthetic": True}),
                "medication_event_build_run_id": "medication-events-run",
                "medication_event_contract_version": MEDICATION_EVENTS_CONTRACT_VERSION,
            }
        )
        row.update(overrides)
        records.append(row)
    return pd.DataFrame(records, columns=MEDICATION_EVENT_COLUMNS)


def make_encounter_index() -> pd.DataFrame:
    """Build a minimally valid encounter-index artifact."""
    row = {column_name: None for column_name in ENCOUNTER_INDEX_COLUMNS}
    row.update(
        {
            "subject_id": 101,
            "sex": "F",
            "age_proxy": 75,
            "age_group": "75-84",
            "encounter_id": "hadm:1",
            "hadm_id": 1,
            "stay_id": 11,
            "encounter_source": "hospital_only",
            "linked_ed_stay_flag": 0,
            "linked_hospital_admission_flag": 1,
            "admission_type": "ELECTIVE",
            "admittime": "2125-03-20 07:00:00",
            "dischtime": "2125-03-20 18:00:00",
            "hospital_length_of_stay_days": 0.458,
            "intime": None,
            "outtime": None,
            "ed_length_of_stay_hours": None,
            "ed_disposition": None,
            "arrival_transport": None,
            "encounter_start": "2125-03-20 07:00:00",
            "encounter_end": "2125-03-20 18:00:00",
            "source_tables_json": json.dumps(["admissions"]),
            "source_record_provenance_json": json.dumps({"synthetic": True}),
            "encounter_index_build_run_id": "encounter-index-run",
            "encounter_index_contract_version": ENCOUNTER_INDEX_CONTRACT_VERSION,
        }
    )
    return pd.DataFrame([row], columns=ENCOUNTER_INDEX_COLUMNS)


if __name__ == "__main__":
    unittest.main()
