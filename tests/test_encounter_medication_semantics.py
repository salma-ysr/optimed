"""Tests for first-pass encounter-medication semantics and burden artifacts."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import (
    ENCOUNTER_INDEX_COLUMNS,
    ENCOUNTER_INDEX_CONTRACT_VERSION,
    MEDICATION_EVENT_COLUMNS,
    MEDICATION_EVENTS_CONTRACT_VERSION,
    MEDICATION_RXNORM_MAPPING_COLUMNS,
    MEDICATION_RXNORM_MAPPING_CONTRACT_VERSION,
)
from opti_med.data_access.encounter_medication_semantics import (
    build_encounter_medication_burden,
    build_encounter_medication_semantics,
    calculate_first_scope_qc_metrics,
    summarize_first_scope_artifacts,
)
from opti_med.data_access.encounter_medication_state import build_encounter_medication_state


class EncounterMedicationSemanticsTests(unittest.TestCase):
    def test_state_and_semantics_populate_first_scope_classes_from_mapping(self) -> None:
        encounter_index = _encounter_index_artifact()
        medication_events = _medication_events_artifact()
        medication_rxnorm_mapping = _medication_rxnorm_mapping_artifact()

        state = build_encounter_medication_state(
            encounter_index=encounter_index,
            medication_events=medication_events,
            medication_rxnorm_mapping=medication_rxnorm_mapping,
            review_time_policy="discharge_capped_latest_available",
        )
        semantics = build_encounter_medication_semantics(
            encounter_medication_state=state,
            medication_rxnorm_mapping=medication_rxnorm_mapping,
        )

        classes = {
            row["medication_standardized"]: row["medication_class_standardized"]
            for row in semantics.to_dict(orient="records")
        }
        self.assertEqual(classes["Lorazepam"], "benzodiazepine")
        self.assertEqual(classes["Diazepam"], "benzodiazepine")
        self.assertEqual(classes["Pantoprazole"], "ppi")
        self.assertEqual(classes["Oxycodone"], "opioid")
        self.assertEqual(classes["Quetiapine"], "antipsychotic")
        self.assertEqual(classes["Furosemide"], "unresolved")
        self.assertEqual(
            semantics.loc[
                semantics["medication_standardized"] == "Furosemide",
                "class_assignment_status",
            ].item(),
            "supported_scope_unresolved",
        )
        self.assertEqual(
            semantics.loc[
                semantics["medication_standardized"] == "Lorazepam",
                "benzodiazepine_heuristic_flag",
            ].item(),
            1,
        )

    def test_burden_features_capture_current_counts_duplicates_schedule_duration_and_placeholders(
        self,
    ) -> None:
        encounter_index = _encounter_index_artifact()
        medication_events = _medication_events_artifact()
        medication_rxnorm_mapping = _medication_rxnorm_mapping_artifact()
        labevents = pd.DataFrame(
            [
                {
                    "hadm_id": 10,
                    "itemid": 50912,
                    "charttime": "2125-03-20 08:00:00",
                    "valuenum": 1.4,
                }
            ]
        )

        state = build_encounter_medication_state(
            encounter_index=encounter_index,
            medication_events=medication_events,
            medication_rxnorm_mapping=medication_rxnorm_mapping,
            review_time_policy="discharge_capped_latest_available",
        )
        semantics = build_encounter_medication_semantics(
            encounter_medication_state=state,
            medication_rxnorm_mapping=medication_rxnorm_mapping,
        )
        burden = build_encounter_medication_burden(
            encounter_medication_semantics=semantics,
            medication_events=medication_events,
            medication_rxnorm_mapping=medication_rxnorm_mapping,
            encounter_index=encounter_index,
            labevents=labevents,
            settings=Settings(),
        )

        lorazepam = burden.loc[burden["medication_standardized"] == "Lorazepam"].iloc[0]
        diazepam = burden.loc[burden["medication_standardized"] == "Diazepam"].iloc[0]
        oxycodone = burden.loc[burden["medication_standardized"] == "Oxycodone"].iloc[0]
        pantoprazole = burden.loc[burden["medication_standardized"] == "Pantoprazole"].iloc[0]
        quetiapine = burden.loc[burden["medication_standardized"] == "Quetiapine"].iloc[0]
        furosemide = burden.loc[burden["medication_standardized"] == "Furosemide"].iloc[0]

        self.assertEqual(int(lorazepam["exact_current_medication_count"]), 5)
        self.assertEqual(int(furosemide["exact_current_medication_count"]), 5)
        self.assertEqual(int(lorazepam["current_benzodiazepine_count"]), 2)
        self.assertEqual(int(oxycodone["current_opioid_count"]), 1)
        self.assertEqual(int(pantoprazole["current_ppi_count"]), 1)
        self.assertEqual(int(quetiapine["current_antipsychotic_count"]), 0)
        self.assertEqual(int(lorazepam["same_class_duplicate_therapy_flag"]), 1)
        self.assertEqual(int(diazepam["same_class_duplicate_therapy_flag"]), 1)
        self.assertEqual(int(lorazepam["same_class_duplicate_therapy_signal_count"]), 2)
        self.assertEqual(int(furosemide["same_class_duplicate_therapy_flag"]), 0)
        self.assertEqual(lorazepam["medication_start_context"], "continued_from_home")
        self.assertEqual(diazepam["medication_start_context"], "new_start_during_encounter")
        self.assertEqual(lorazepam["scheduled_vs_prn"], "prn")
        self.assertEqual(pantoprazole["scheduled_vs_prn"], "scheduled")
        self.assertEqual(int(lorazepam["duration_before_review_inferable_flag"]), 1)
        self.assertEqual(int(lorazepam["duration_before_review_lower_bound_flag"]), 1)
        self.assertAlmostEqual(float(lorazepam["duration_before_review_hours"]), 15.5, places=3)
        self.assertEqual(oxycodone["opioid_mme_status"], "ready_for_future_conversion")
        self.assertEqual(int(oxycodone["opioid_mme_required_fields_present_flag"]), 1)
        self.assertEqual(
            oxycodone["renal_dose_mismatch_status"],
            "ready_for_future_logic",
        )
        self.assertEqual(int(oxycodone["renal_context_available_flag"]), 1)
        self.assertEqual(int(oxycodone["renal_dose_fields_available_flag"]), 1)
        self.assertEqual(int(quetiapine["active_at_review_flag"]), 0)
        self.assertEqual(
            quetiapine["renal_dose_mismatch_status"],
            "not_current_medication",
        )

        metrics = calculate_first_scope_qc_metrics(
            encounter_medication_semantics=semantics,
            encounter_medication_burden=burden,
        )
        self.assertEqual(metrics["class_counts_active"]["benzodiazepine"], 2)
        self.assertEqual(metrics["class_counts_active"]["unresolved"], 1)
        self.assertEqual(metrics["exact_duplicate_therapy_signal_groups"], 1)
        self.assertEqual(metrics["exact_duplicate_therapy_signal_rows"], 2)
        summary_lines = summarize_first_scope_artifacts(
            encounter_medication_semantics=semantics,
            encounter_medication_burden=burden,
        )
        self.assertTrue(any("unresolved_class_rate=" in line for line in summary_lines))
        self.assertTrue(any("heuristic_comparison benzodiazepine" in line for line in summary_lines))
        self.assertTrue(any("exact_duplicate_therapy_signals=" in line for line in summary_lines))


def _encounter_index_artifact() -> pd.DataFrame:
    row = {
        "subject_id": 1,
        "sex": "F",
        "age_proxy": 77,
        "age_group": "75-84",
        "encounter_id": "hadm:10",
        "hadm_id": 10,
        "stay_id": pd.NA,
        "encounter_source": "hospital_only",
        "linked_ed_stay_flag": 0,
        "linked_hospital_admission_flag": 1,
        "admission_type": "URGENT",
        "admittime": "2125-03-19 18:00:00",
        "dischtime": "2125-03-20 10:00:00",
        "hospital_length_of_stay_days": 0.667,
        "intime": pd.NA,
        "outtime": pd.NA,
        "ed_length_of_stay_hours": pd.NA,
        "ed_disposition": pd.NA,
        "arrival_transport": pd.NA,
        "encounter_start": "2125-03-19 18:00:00",
        "encounter_end": "2125-03-20 10:00:00",
        "source_tables_json": "[\"admissions\", \"patients\"]",
        "source_record_provenance_json": "{\"roles\": [\"admissions\", \"patients\"]}",
        "encounter_index_build_run_id": "encounter-index-test",
        "encounter_index_contract_version": ENCOUNTER_INDEX_CONTRACT_VERSION,
    }
    return pd.DataFrame([{column_name: row.get(column_name) for column_name in ENCOUNTER_INDEX_COLUMNS}])


def _medication_events_artifact() -> pd.DataFrame:
    rows = [
        _medication_event_row(
            medication_event_id="home-lorazepam",
            medication_event_type="home_medrecon",
            event_source_category="home_medication_reconciliation",
            event_source_table="medrecon",
            raw_medication_name="Lorazepam 0.5 mg",
            medication_normalized="lorazepam",
            event_time="2125-03-19 18:30:00",
            starttime="2125-03-19 18:30:00",
            stoptime=pd.NA,
            route=pd.NA,
            frequency=pd.NA,
            status=pd.NA,
            dose_value=pd.NA,
            dose_unit=pd.NA,
            source_home_medrecon=1,
            source_ed_pyxis=0,
            source_hospital_order=0,
            source_hospital_admin=0,
            continued_from_home_inferred=1,
            newly_started_during_encounter_inferred=0,
        ),
        _medication_event_row(
            medication_event_id="order-lorazepam",
            medication_event_type="hospital_order",
            event_source_category="hospital_medication_order",
            event_source_table="prescriptions",
            raw_medication_name="Lorazepam 0.5 mg",
            medication_normalized="lorazepam",
            event_time="2125-03-19 19:00:00",
            starttime="2125-03-19 19:00:00",
            stoptime="2125-03-20 12:00:00",
            route="PO",
            frequency="Q6H PRN",
            status="active",
            dose_value="0.5",
            dose_unit="mg",
            source_home_medrecon=0,
            source_ed_pyxis=0,
            source_hospital_order=1,
            source_hospital_admin=0,
            continued_from_home_inferred=1,
            newly_started_during_encounter_inferred=0,
            pharmacy_id=500,
            poe_id=700,
        ),
        _medication_event_row(
            medication_event_id="order-diazepam",
            medication_event_type="hospital_order",
            event_source_category="hospital_medication_order",
            event_source_table="prescriptions",
            raw_medication_name="Diazepam 5 mg",
            medication_normalized="diazepam",
            event_time="2125-03-19 20:00:00",
            starttime="2125-03-19 20:00:00",
            stoptime="2125-03-20 12:00:00",
            route="PO",
            frequency="BID",
            status="active",
            dose_value="5",
            dose_unit="mg",
            source_home_medrecon=0,
            source_ed_pyxis=0,
            source_hospital_order=1,
            source_hospital_admin=0,
            continued_from_home_inferred=0,
            newly_started_during_encounter_inferred=1,
            pharmacy_id=501,
            poe_id=701,
        ),
        _medication_event_row(
            medication_event_id="order-pantoprazole",
            medication_event_type="hospital_order",
            event_source_category="hospital_medication_order",
            event_source_table="prescriptions",
            raw_medication_name="Pantoprazole 40 mg",
            medication_normalized="pantoprazole",
            event_time="2125-03-19 21:00:00",
            starttime="2125-03-19 21:00:00",
            stoptime="2125-03-20 12:00:00",
            route="PO",
            frequency="DAILY",
            status="active",
            dose_value="40",
            dose_unit="mg",
            source_home_medrecon=0,
            source_ed_pyxis=0,
            source_hospital_order=1,
            source_hospital_admin=0,
            continued_from_home_inferred=0,
            newly_started_during_encounter_inferred=1,
            pharmacy_id=502,
            poe_id=702,
        ),
        _medication_event_row(
            medication_event_id="order-oxycodone",
            medication_event_type="hospital_order",
            event_source_category="hospital_medication_order",
            event_source_table="prescriptions",
            raw_medication_name="Oxycodone 5 mg",
            medication_normalized="oxycodone",
            event_time="2125-03-19 22:00:00",
            starttime="2125-03-19 22:00:00",
            stoptime="2125-03-20 12:00:00",
            route="PO",
            frequency="Q4H",
            status="active",
            dose_value="5",
            dose_unit="mg",
            source_home_medrecon=0,
            source_ed_pyxis=0,
            source_hospital_order=1,
            source_hospital_admin=0,
            continued_from_home_inferred=0,
            newly_started_during_encounter_inferred=1,
            pharmacy_id=503,
            poe_id=703,
        ),
        _medication_event_row(
            medication_event_id="order-furosemide",
            medication_event_type="hospital_order",
            event_source_category="hospital_medication_order",
            event_source_table="prescriptions",
            raw_medication_name="Furosemide 20 mg",
            medication_normalized="furosemide",
            event_time="2125-03-19 23:00:00",
            starttime="2125-03-19 23:00:00",
            stoptime="2125-03-20 12:00:00",
            route="PO",
            frequency="BID",
            status="active",
            dose_value="20",
            dose_unit="mg",
            source_home_medrecon=0,
            source_ed_pyxis=0,
            source_hospital_order=1,
            source_hospital_admin=0,
            continued_from_home_inferred=0,
            newly_started_during_encounter_inferred=1,
            pharmacy_id=504,
            poe_id=704,
        ),
        _medication_event_row(
            medication_event_id="order-quetiapine",
            medication_event_type="hospital_order",
            event_source_category="hospital_medication_order",
            event_source_table="prescriptions",
            raw_medication_name="Quetiapine 25 mg",
            medication_normalized="quetiapine",
            event_time="2125-03-19 19:15:00",
            starttime="2125-03-19 19:15:00",
            stoptime="2125-03-20 09:00:00",
            route="PO",
            frequency="HS",
            status="expired",
            dose_value="25",
            dose_unit="mg",
            source_home_medrecon=0,
            source_ed_pyxis=0,
            source_hospital_order=1,
            source_hospital_admin=0,
            continued_from_home_inferred=0,
            newly_started_during_encounter_inferred=1,
            pharmacy_id=505,
            poe_id=705,
        ),
    ]
    return pd.DataFrame(rows).loc[:, MEDICATION_EVENT_COLUMNS]


def _medication_event_row(
    *,
    medication_event_id: str,
    medication_event_type: str,
    event_source_category: str,
    event_source_table: str,
    raw_medication_name: str,
    medication_normalized: str,
    event_time: object,
    starttime: object,
    stoptime: object,
    route: object,
    frequency: object,
    status: object,
    dose_value: object,
    dose_unit: object,
    source_home_medrecon: int,
    source_ed_pyxis: int,
    source_hospital_order: int,
    source_hospital_admin: int,
    continued_from_home_inferred: int,
    newly_started_during_encounter_inferred: int,
    pharmacy_id: object = pd.NA,
    poe_id: object = pd.NA,
) -> dict[str, object]:
    row = {
        "subject_id": 1,
        "hadm_id": 10,
        "stay_id": pd.NA,
        "encounter_id": "hadm:10",
        "encounter_source": "hospital_only",
        "encounter_start": "2125-03-19 18:00:00",
        "encounter_end": "2125-03-20 10:00:00",
        "medication_event_id": medication_event_id,
        "medication_event_type": medication_event_type,
        "event_source_category": event_source_category,
        "event_source_table": event_source_table,
        "raw_medication_name": raw_medication_name,
        "medication_name": raw_medication_name,
        "medication_normalized": medication_normalized,
        "medication_prestandardized_text": medication_normalized,
        "event_time": event_time,
        "starttime": starttime,
        "stoptime": stoptime,
        "route": route,
        "frequency": frequency,
        "status": status,
        "dose_value": dose_value,
        "dose_unit": dose_unit,
        "pharmacy_id": pharmacy_id,
        "poe_id": poe_id,
        "emar_id": pd.NA,
        "emar_seq": pd.NA,
        "source_home_medrecon": source_home_medrecon,
        "source_ed_pyxis": source_ed_pyxis,
        "source_hospital_order": source_hospital_order,
        "source_hospital_admin": source_hospital_admin,
        "pharmacy_enriched_flag": int(source_hospital_order == 1),
        "order_enrichment_applied_flag": int(source_hospital_order == 1),
        "order_enrichment_source_table": "pharmacy" if source_hospital_order == 1 else pd.NA,
        "continued_from_home_inferred": continued_from_home_inferred,
        "continued_from_home_inferred_flag": continued_from_home_inferred,
        "newly_started_during_encounter_inferred": newly_started_during_encounter_inferred,
        "newly_started_during_encounter_inferred_flag": newly_started_during_encounter_inferred,
        "continuity_inference_rule": "synthetic_test_rule" if continued_from_home_inferred == 1 else pd.NA,
        "medication_episode_id": f"episode-{medication_normalized}",
        "prescription_segment_count": 1,
        "prescription_segments_json": "[]",
        "source_tables_json": f"[\"{event_source_table}\"]",
        "source_record_provenance_json": f"{{\"role\": \"{event_source_table}\"}}",
        "medication_event_build_run_id": "medication-events-test",
        "medication_event_contract_version": MEDICATION_EVENTS_CONTRACT_VERSION,
    }
    return {column_name: row.get(column_name) for column_name in MEDICATION_EVENT_COLUMNS}


def _medication_rxnorm_mapping_artifact() -> pd.DataFrame:
    rows = [
        _mapping_row("lorazepam", "Lorazepam", "benzodiazepine"),
        _mapping_row("diazepam", "Diazepam", "benzodiazepine"),
        _mapping_row("pantoprazole", "Pantoprazole", "ppi"),
        _mapping_row("oxycodone", "Oxycodone", "opioid"),
        _mapping_row("quetiapine", "Quetiapine", "antipsychotic"),
        _mapping_row("furosemide", "Furosemide", None),
    ]
    return pd.DataFrame(rows).loc[:, MEDICATION_RXNORM_MAPPING_COLUMNS]


def _mapping_row(
    medication_query_key: str,
    ingredient_standardized: str,
    supported_class_label: str | None,
) -> dict[str, object]:
    class_labels = [] if supported_class_label is None else [supported_class_label]
    row = {
        "medication_query_key": medication_query_key,
        "raw_medication_string": ingredient_standardized,
        "normalized_query_string": medication_query_key,
        "lookup_mode": "cache_first",
        "lookup_strategy_used": "cached_prior_result",
        "lookup_status": "resolved",
        "rxnorm_rxcui": f"rxcui-{medication_query_key}",
        "matched_term": ingredient_standardized,
        "matched_term_type": "IN",
        "ingredient_rxcui": f"rxcui-{medication_query_key}",
        "ingredient_standardized": ingredient_standardized,
        "ingredient_resolution_status": "resolved_to_ingredient",
        "medication_standardized": ingredient_standardized,
        "medication_standardized_source": "rxnorm_ingredient",
        "mapping_confidence": "high",
        "ambiguous_match_flag": 0,
        "candidate_match_count": 1,
        "candidate_rxcuis_json": dumps_json([f"rxcui-{medication_query_key}"]),
        "ambiguity_note": None,
        "class_assignment_status": (
            "supported_scope_class_assigned"
            if supported_class_label is not None
            else "supported_scope_unresolved"
        ),
        "class_ids_json": dumps_json(
            [] if supported_class_label is None else [f"opti_med_first_scope:{supported_class_label}"]
        ),
        "class_labels_json": dumps_json(class_labels),
        "lookup_timestamp": "2125-03-20 10:00:00",
        "api_called_flag": 0,
        "mapper_name": "rxnorm_api_bootstrap_mapper",
        "mapper_version": "0.1",
        "rxnorm_version": "2026-03-02",
        "rxnorm_api_version": "rxnav_rest_v1",
        "mapping_provenance_json": "{\"lookup_mode\": \"cache_first\"}",
        "medication_rxnorm_mapping_build_run_id": "medication-rxnorm-mapping-test",
        "medication_rxnorm_mapping_contract_version": MEDICATION_RXNORM_MAPPING_CONTRACT_VERSION,
    }
    return {column_name: row.get(column_name) for column_name in MEDICATION_RXNORM_MAPPING_COLUMNS}


def dumps_json(values: object) -> str:
    return json.dumps(values, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
