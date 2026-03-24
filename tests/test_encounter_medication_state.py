"""Tests for the encounter-medication-state builder and QC helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from opti_med.data_access.encounter_medication_state import (
    build_encounter_medication_state,
    calculate_encounter_medication_state_qc_metrics,
    resolve_policy_safe_review_timestamp_for_encounter,
)


class EncounterMedicationStateTests(unittest.TestCase):
    def test_review_timestamp_is_capped_to_discharge_when_policy_requires(self) -> None:
        encounter = {
            "subject_id": 1,
            "hadm_id": 10,
            "stay_id": None,
            "encounter_id": "hadm:10",
            "admittime": "2125-03-19 18:00:00",
            "dischtime": "2125-03-20 10:00:00",
            "encounter_start": "2125-03-19 18:00:00",
            "encounter_end": "2125-03-20 10:00:00",
        }
        medication_events = pd.DataFrame(
            [
                {
                    "subject_id": 1,
                    "encounter_id": "hadm:10",
                    "medication_event_type": "hospital_order",
                    "event_time": "2125-03-19 19:00:00",
                    "starttime": "2125-03-19 19:00:00",
                    "stoptime": "2125-03-20 12:00:00",
                }
            ]
        )

        review_resolution = resolve_policy_safe_review_timestamp_for_encounter(
            encounter=encounter,
            medication_events=medication_events,
            review_time_policy="discharge_capped_latest_available",
        )

        self.assertEqual(
            str(review_resolution.review_timestamp),
            "2125-03-20 10:00:00",
        )
        self.assertEqual(review_resolution.review_timestamp_source, "encounter_boundary")
        self.assertEqual(
            str(review_resolution.review_timestamp_candidate),
            "2125-03-20 12:00:00",
        )
        self.assertEqual(
            review_resolution.review_timestamp_candidate_source,
            "medication_order",
        )
        self.assertTrue(review_resolution.review_time_capped_to_discharge_flag)

    def test_builder_preserves_active_and_inactive_candidates_at_review_time(self) -> None:
        encounter_index = pd.DataFrame(
            [
                {
                    "subject_id": 1,
                    "encounter_id": "hadm:10",
                    "hadm_id": 10,
                    "stay_id": None,
                    "encounter_start": "2125-03-19 18:00:00",
                    "encounter_end": "2125-03-20 10:00:00",
                    "admittime": "2125-03-19 18:00:00",
                    "dischtime": "2125-03-20 10:00:00",
                    "intime": None,
                    "outtime": None,
                }
            ]
        )
        medication_events = pd.DataFrame(
            [
                {
                    "subject_id": 1,
                    "hadm_id": 10,
                    "stay_id": None,
                    "encounter_id": "hadm:10",
                    "encounter_source": "hospital_only",
                    "encounter_start": "2125-03-19 18:00:00",
                    "encounter_end": "2125-03-20 10:00:00",
                    "medication_event_id": "order-furosemide",
                    "medication_event_type": "hospital_order",
                    "event_source_table": "prescriptions",
                    "raw_medication_name": "Furosemide 20 mg",
                    "medication_name": "Furosemide",
                    "medication_normalized": "furosemide",
                    "event_time": "2125-03-19 19:00:00",
                    "starttime": "2125-03-19 19:00:00",
                    "stoptime": "2125-03-20 12:00:00",
                    "route": "PO",
                    "frequency": "BID",
                    "status": "active",
                    "continued_from_home_inferred": 0,
                    "newly_started_during_encounter_inferred": 1,
                    "source_home_medrecon": 0,
                    "source_ed_pyxis": 0,
                    "source_hospital_order": 1,
                    "source_hospital_admin": 0,
                    "pharmacy_enriched_flag": 1,
                    "source_tables_json": "[\"prescriptions\"]",
                    "source_record_provenance_json": "{\"role\": \"prescriptions\"}",
                },
                {
                    "subject_id": 1,
                    "hadm_id": 10,
                    "stay_id": None,
                    "encounter_id": "hadm:10",
                    "encounter_source": "hospital_only",
                    "encounter_start": "2125-03-19 18:00:00",
                    "encounter_end": "2125-03-20 10:00:00",
                    "medication_event_id": "order-midazolam",
                    "medication_event_type": "hospital_order",
                    "event_source_table": "prescriptions",
                    "raw_medication_name": "Midazolam",
                    "medication_name": "Midazolam",
                    "medication_normalized": "midazolam",
                    "event_time": "2125-03-19 19:30:00",
                    "starttime": "2125-03-19 19:30:00",
                    "stoptime": "2125-03-20 09:30:00",
                    "route": "IV",
                    "frequency": "ONCE",
                    "status": "expired",
                    "continued_from_home_inferred": 0,
                    "newly_started_during_encounter_inferred": 1,
                    "source_home_medrecon": 0,
                    "source_ed_pyxis": 0,
                    "source_hospital_order": 1,
                    "source_hospital_admin": 0,
                    "pharmacy_enriched_flag": 1,
                    "source_tables_json": "[\"prescriptions\"]",
                    "source_record_provenance_json": "{\"role\": \"prescriptions\"}",
                },
            ]
        )

        encounter_medication_state = build_encounter_medication_state(
            encounter_index=encounter_index,
            medication_events=medication_events,
            review_time_policy="discharge_capped_latest_available",
        )

        statuses = {
            row["medication_standardized"]: row["medication_status_at_review"]
            for row in encounter_medication_state.to_dict(orient="records")
        }
        self.assertEqual(len(encounter_medication_state), 2)
        self.assertEqual(statuses["furosemide"], "active_at_review_time")
        self.assertEqual(statuses["midazolam"], "inactive_before_review_time")
        self.assertEqual(
            encounter_medication_state.loc[
                encounter_medication_state["medication_standardized"] == "furosemide",
                "active_at_review_flag",
            ].item(),
            1,
        )
        self.assertEqual(
            encounter_medication_state.loc[
                encounter_medication_state["medication_standardized"] == "midazolam",
                "active_at_review_flag",
            ].item(),
            0,
        )
        self.assertTrue(
            (
                pd.to_datetime(
                    encounter_medication_state["review_timestamp"],
                    errors="coerce",
                )
                <= pd.to_datetime(
                    encounter_medication_state["discharge_boundary"],
                    errors="coerce",
                )
            ).all()
        )

    def test_qc_metrics_report_duplicate_keys_and_uncertain_states(self) -> None:
        dataframe = pd.DataFrame(
            [
                {
                    "subject_id": 1,
                    "encounter_id": "hadm:10",
                    "review_timestamp": "2125-03-20 10:00:00",
                    "discharge_boundary": "2125-03-20 10:00:00",
                    "medication_standardized": "furosemide",
                    "medication_status_at_review": "activity_uncertain_at_review_time",
                },
                {
                    "subject_id": 1,
                    "encounter_id": "hadm:10",
                    "review_timestamp": "2125-03-20 10:00:00",
                    "discharge_boundary": "2125-03-20 10:00:00",
                    "medication_standardized": "furosemide",
                    "medication_status_at_review": "activity_uncertain_at_review_time",
                },
            ]
        )

        metrics = calculate_encounter_medication_state_qc_metrics(dataframe)

        self.assertEqual(metrics["duplicate_key_count"], 2)
        self.assertEqual(metrics["uncertain_state_count"], 2)
        self.assertEqual(metrics["review_timestamp_after_discharge_count"], 0)
        self.assertEqual(
            metrics["counts_by_medication_status_at_review"],
            {"activity_uncertain_at_review_time": 2},
        )


if __name__ == "__main__":
    unittest.main()
