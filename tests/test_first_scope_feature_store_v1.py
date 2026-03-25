"""Tests for first-scope Feature Store v1 staged builders."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from opti_med.data_access.artifact_schemas import (
    ENCOUNTER_MEDICATION_BURDEN_CONTRACT_VERSION,
    ENCOUNTER_MEDICATION_FIRST_SCOPE_COLUMNS,
    ENCOUNTER_MEDICATION_FIRST_SCOPE_CONTRACT_VERSION,
    ENCOUNTER_MEDICATION_SEMANTICS_CONTRACT_VERSION,
)
from opti_med.features.first_scope_feature_store import (
    STATE_PROVENANCE_COLUMNS,
    build_first_scope_feature_store,
    build_first_scope_patient_context_features,
    build_first_scope_temporal_features,
)


def make_first_scope(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a minimally valid synthetic first-scope artifact."""
    records: list[dict[str, object]] = []
    for index, overrides in enumerate(rows):
        row = {column_name: None for column_name in ENCOUNTER_MEDICATION_FIRST_SCOPE_COLUMNS}
        row.update(
            {
                "subject_id": 101,
                "encounter_id": "hadm:1",
                "hadm_id": 1,
                "stay_id": 11,
                "review_timestamp": "2125-03-20 10:00:00",
                "review_timestamp_source": "medication_administration",
                "age_proxy": 75,
                "age_group": "75-84",
                "medication_standardized": f"morphine-{index}",
                "medication_standardized_source": "rxnorm_ingredient",
                "medication_normalized": "morphine",
                "medication_raw": "Morphine",
                "rxnorm_rxcui": "7052",
                "rxnorm_matched_term": "morphine",
                "rxnorm_term_type": "IN",
                "ingredient_standardized": "morphine",
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
                "selected_medication_event_id": f"event-{index}",
                "selected_medication_event_type": "hospital_order",
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
                "scheduled_vs_prn": "scheduled",
                "scheduled_vs_prn_inference_status": "explicit_scheduled",
                "dose_value": "5",
                "dose_unit": "mg",
                "route": "PO",
                "frequency": "BID",
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


def make_state_provenance_subset(first_scope: pd.DataFrame) -> pd.DataFrame:
    """Build a minimally valid keyed state-provenance subset."""
    records: list[dict[str, object]] = []
    for row in first_scope.to_dict(orient="records"):
        records.append(
            {
                "subject_id": row["subject_id"],
                "encounter_id": row["encounter_id"],
                "medication_standardized": row["medication_standardized"],
                "review_timestamp": row["review_timestamp"],
                "review_time_validated_flag": 1,
                "review_time_policy_name": "discharge_capped_latest_available",
                "review_time_capped_to_discharge_flag": 0,
                "source_home_medrecon_flag": 0,
                "source_ed_pyxis_flag": 0,
                "source_hospital_order_flag": 1,
                "source_hospital_admin_flag": 0,
                "source_tables_json": json.dumps(["pharmacy", "prescriptions"]),
            }
        )
    return pd.DataFrame(records, columns=STATE_PROVENANCE_COLUMNS)


class FirstScopeTemporalFeatureTests(unittest.TestCase):
    def test_temporal_lab_summary_excludes_post_review_observations(self) -> None:
        first_scope = make_first_scope([{}])
        labevents = pd.DataFrame(
            [
                {"hadm_id": 1, "itemid": 50912, "charttime": "2125-03-20 08:00:00", "valuenum": 1.0},
                {"hadm_id": 1, "itemid": 50912, "charttime": "2125-03-20 09:00:00", "valuenum": 1.4},
                {"hadm_id": 1, "itemid": 50912, "charttime": "2125-03-20 11:00:00", "valuenum": 2.0},
            ]
        )

        temporal = build_first_scope_temporal_features(
            encounter_medication_first_scope=first_scope,
            labevents=labevents,
            serum_creatinine_item_ids=(50912,),
        )

        row = temporal.iloc[0]
        self.assertEqual(row["creatinine_observation_count_pre_review"], 2)
        self.assertEqual(row["creatinine_first"], 1.0)
        self.assertEqual(row["creatinine_last"], 1.4)
        self.assertEqual(row["creatinine_max"], 1.4)
        self.assertEqual(str(row["creatinine_latest_time"]), "2125-03-20 09:00:00")
        self.assertEqual(row["lab_window_end_at_or_before_review_time_flag"], 1)

    def test_temporal_vitals_exclude_post_review_vitals(self) -> None:
        first_scope = make_first_scope([{}])
        edstays = pd.DataFrame(
            [
                {
                    "subject_id": 101,
                    "stay_id": 11,
                    "intime": "2125-03-20 07:00:00",
                }
            ]
        )
        triage = pd.DataFrame(
            [
                {
                    "subject_id": 101,
                    "stay_id": 11,
                    "charttime": "2125-03-20 07:30:00",
                    "sbp": 100,
                    "dbp": 70,
                    "heartrate": 80,
                    "pain": 0,
                }
            ]
        )
        vitalsign = pd.DataFrame(
            [
                {
                    "subject_id": 101,
                    "stay_id": 11,
                    "charttime": "2125-03-20 09:00:00",
                    "sbp": 110,
                    "dbp": 75,
                    "heartrate": 90,
                    "pain": 4,
                },
                {
                    "subject_id": 101,
                    "stay_id": 11,
                    "charttime": "2125-03-20 10:30:00",
                    "sbp": 140,
                    "dbp": 95,
                    "heartrate": 130,
                    "pain": 8,
                },
            ]
        )

        temporal = build_first_scope_temporal_features(
            encounter_medication_first_scope=first_scope,
            labevents=pd.DataFrame(columns=["hadm_id", "itemid", "charttime", "valuenum"]),
            serum_creatinine_item_ids=(50912,),
            edstays=edstays,
            triage=triage,
            vitalsign=vitalsign,
        )

        row = temporal.iloc[0]
        self.assertEqual(row["sbp_last"], 110)
        self.assertEqual(row["heart_rate_last"], 90)
        self.assertEqual(row["vital_window_end_at_or_before_review_time_flag"], 1)
        self.assertEqual(str(row["vital_window_end"]), "2125-03-20 09:00:00")
        self.assertAlmostEqual(row["heart_rate_mean"], 85.0)


class FirstScopeFeatureStoreJoinTests(unittest.TestCase):
    def test_final_feature_store_preserves_first_scope_rows(self) -> None:
        first_scope = make_first_scope(
            [
                {},
                {
                    "medication_standardized": "oxycodone",
                    "ingredient_standardized": "oxycodone",
                    "rxnorm_rxcui": "7804",
                    "rxnorm_matched_term": "oxycodone",
                    "medication_normalized": "oxycodone",
                    "medication_raw": "Oxycodone",
                },
            ]
        )
        patients = pd.DataFrame([{"subject_id": 101, "gender": "F"}])
        admissions = pd.DataFrame(
            [
                {"subject_id": 101, "hadm_id": 900, "dischtime": "2125-03-10 10:00:00"},
                {"subject_id": 101, "hadm_id": 1, "dischtime": "2125-03-22 10:00:00"},
            ]
        )
        diagnoses_icd = pd.DataFrame(
            [
                {"subject_id": 101, "hadm_id": 900, "icd_code": "N18", "icd_version": 10},
            ]
        )
        labevents = pd.DataFrame(
            [
                {"hadm_id": 1, "itemid": 50912, "charttime": "2125-03-20 09:00:00", "valuenum": 1.2},
                {"hadm_id": 1, "itemid": 50912, "charttime": "2125-03-20 11:00:00", "valuenum": 2.1},
            ]
        )
        omr = pd.DataFrame(
            [
                {"subject_id": 101, "chartdate": "2125-03-18", "result_name": "Weight (Lbs)", "result_value": "150"},
                {"subject_id": 101, "chartdate": "2125-03-18", "result_name": "Height (Inches)", "result_value": "66"},
            ]
        )
        edstays = pd.DataFrame(
            [
                {
                    "subject_id": 101,
                    "stay_id": 11,
                    "intime": "2125-03-20 07:00:00",
                    "outtime": "2125-03-20 09:30:00",
                }
            ]
        )
        ed_diagnosis = pd.DataFrame(
            [
                {"subject_id": 101, "stay_id": 11, "icd_code": "F03", "icd_version": 10},
            ]
        )
        triage = pd.DataFrame(
            [
                {
                    "subject_id": 101,
                    "stay_id": 11,
                    "charttime": "2125-03-20 07:30:00",
                    "sbp": 105,
                    "dbp": 72,
                    "heartrate": 82,
                    "pain": 2,
                }
            ]
        )

        patient_context = build_first_scope_patient_context_features(
            encounter_medication_first_scope=first_scope,
            patients=patients,
            admissions=admissions,
            diagnoses_icd=diagnoses_icd,
            labevents=labevents,
            omr=omr,
            edstays=edstays,
            ed_diagnosis=ed_diagnosis,
            serum_creatinine_item_ids=(50912,),
        )
        temporal = build_first_scope_temporal_features(
            encounter_medication_first_scope=first_scope,
            labevents=labevents,
            serum_creatinine_item_ids=(50912,),
            edstays=edstays,
            triage=triage,
            vitalsign=pd.DataFrame(),
        )
        state_subset = make_state_provenance_subset(first_scope)

        result = build_first_scope_feature_store(
            encounter_medication_first_scope=first_scope,
            patient_context_features=patient_context,
            temporal_features=temporal,
            encounter_medication_state_provenance=state_subset,
        )

        feature_store = result.feature_store
        self.assertEqual(len(feature_store), len(first_scope))
        self.assertTrue(
            feature_store["feature_available_as_of_review_time_flag"].eq(1).all()
        )
        self.assertTrue(feature_store["prior_ckd_flag"].eq(1).all())
        self.assertTrue(feature_store["ed_dementia_flag"].eq(1).all())
        self.assertEqual(
            int(feature_store["prior_hospital_admission_count_all_time"].iloc[0]),
            1,
        )
        self.assertAlmostEqual(float(feature_store["weight_kg"].iloc[0]), 68.039, places=3)
        self.assertEqual(
            str(feature_store["creatinine_latest_time"].iloc[0]),
            "2125-03-20 09:00:00",
        )
        missingness_profile = json.loads(feature_store["missingness_profile_json"].iloc[0])
        self.assertIn("creatinine_missing", missingness_profile)
        self.assertFalse(missingness_profile["creatinine_missing"])


if __name__ == "__main__":
    unittest.main()
