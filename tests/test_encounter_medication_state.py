"""Tests for the encounter-medication-state builder and QC helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from opti_med.cohort import OlderAdultEncounterEligibilityBuilder
from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import (
    ENCOUNTER_INDEX_CONTRACT_VERSION,
    MEDICATION_EVENTS_CONTRACT_VERSION,
)
from opti_med.data_access.encounter_medication_state import (
    EncounterMedicationStateArtifactBuilder,
    build_encounter_medication_state,
    calculate_encounter_medication_state_qc_metrics,
    identify_supporting_review_time_encounters,
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
                    "sex": "M",
                    "age_proxy": 77,
                    "age_group": "75-84",
                    "encounter_id": "hadm:10",
                    "hadm_id": 10,
                    "stay_id": None,
                    "encounter_source": "hospital_only",
                    "linked_ed_stay_flag": 0,
                    "linked_hospital_admission_flag": 1,
                    "admission_type": "URGENT",
                    "encounter_start": "2125-03-19 18:00:00",
                    "encounter_end": "2125-03-20 10:00:00",
                    "admittime": "2125-03-19 18:00:00",
                    "dischtime": "2125-03-20 10:00:00",
                    "intime": None,
                    "outtime": None,
                    "hospital_length_of_stay_days": 0.667,
                    "ed_length_of_stay_hours": None,
                    "ed_disposition": None,
                    "arrival_transport": None,
                    "source_tables_json": "[\"patients\", \"admissions\"]",
                    "source_record_provenance_json": "{\"patients\": {\"role\": \"patients\"}, \"admissions\": {\"role\": \"admissions\"}, \"edstays\": null}",
                    "encounter_index_build_run_id": "encounter-index-test",
                    "encounter_index_contract_version": ENCOUNTER_INDEX_CONTRACT_VERSION,
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
        self.assertEqual(encounter_medication_state["age_proxy"].tolist(), [77.0, 77.0])
        self.assertEqual(encounter_medication_state["age_group"].tolist(), ["75-84", "75-84"])
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

    def test_identify_supporting_review_time_encounters_excludes_medication_supported_rows(self) -> None:
        encounter_index = pd.DataFrame(
            [
                {"encounter_id": "hadm:10", "subject_id": 1, "hadm_id": 10, "stay_id": None},
                {"encounter_id": "hadm:11", "subject_id": 1, "hadm_id": 11, "stay_id": None},
            ]
        )
        medication_events = pd.DataFrame(
            [
                {
                    "encounter_id": "hadm:10",
                    "medication_event_type": "hospital_order",
                    "event_time": "2125-03-19 19:00:00",
                    "starttime": "2125-03-19 19:00:00",
                    "stoptime": "2125-03-20 12:00:00",
                },
                {
                    "encounter_id": "hadm:11",
                    "medication_event_type": "home_medrecon",
                    "event_time": "2125-03-19 19:00:00",
                    "starttime": None,
                    "stoptime": None,
                },
            ]
        )

        supporting = identify_supporting_review_time_encounters(
            encounter_index=encounter_index,
            medication_events=medication_events,
        )

        self.assertEqual(supporting["encounter_id"].tolist(), ["hadm:11"])

    def test_artifact_builder_only_expands_eligible_65plus_encounters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            standardized_root = base_dir / "standardized"
            analytical_root = base_dir / "analytical"
            standardized_root.mkdir(parents=True, exist_ok=True)
            analytical_root.mkdir(parents=True, exist_ok=True)

            encounter_index = _encounter_index_for_builder_test()
            medication_events = _medication_events_for_builder_test()
            encounter_index.to_parquet(standardized_root / "encounter_index.parquet", index=False)
            medication_events.to_parquet(standardized_root / "medication_events.parquet", index=False)

            settings = Settings(
                standardized_root=standardized_root,
                analytical_root=analytical_root,
            )
            eligibility_builder = OlderAdultEncounterEligibilityBuilder(settings)
            eligibility_builder.save(eligibility_builder.build(encounter_index=encounter_index))

            encounter_medication_state = EncounterMedicationStateArtifactBuilder(settings).build()

            self.assertEqual(set(encounter_medication_state["encounter_id"].tolist()), {"hadm:10"})
            self.assertEqual(set(encounter_medication_state["subject_id"].tolist()), {1})
            self.assertEqual(set(encounter_medication_state["age_group"].tolist()), {"75-84"})
            self.assertTrue(
                (pd.to_numeric(encounter_medication_state["age_proxy"], errors="coerce") >= 65).all()
            )
            self.assertEqual(
                int(
                    (
                        pd.to_datetime(encounter_medication_state["review_timestamp"], errors="coerce")
                        > pd.to_datetime(encounter_medication_state["discharge_boundary"], errors="coerce")
                    ).sum()
                ),
                0,
            )


def _encounter_index_for_builder_test() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": 1,
                "sex": "M",
                "age_proxy": 77,
                "age_group": "75-84",
                "encounter_id": "hadm:10",
                "hadm_id": 10,
                "stay_id": None,
                "encounter_source": "hospital_only",
                "linked_ed_stay_flag": 0,
                "linked_hospital_admission_flag": 1,
                "admission_type": "URGENT",
                "admittime": "2125-03-19 18:00:00",
                "dischtime": "2125-03-20 10:00:00",
                "hospital_length_of_stay_days": 0.667,
                "intime": None,
                "outtime": None,
                "ed_length_of_stay_hours": None,
                "ed_disposition": None,
                "arrival_transport": None,
                "encounter_start": "2125-03-19 18:00:00",
                "encounter_end": "2125-03-20 10:00:00",
                "source_tables_json": "[\"patients\", \"admissions\"]",
                "source_record_provenance_json": "{\"patients\": {\"role\": \"patients\"}, \"admissions\": {\"role\": \"admissions\"}, \"edstays\": null}",
                "encounter_index_build_run_id": "encounter-index-test",
                "encounter_index_contract_version": ENCOUNTER_INDEX_CONTRACT_VERSION,
            },
            {
                "subject_id": 2,
                "sex": "F",
                "age_proxy": 60,
                "age_group": "<65",
                "encounter_id": "hadm:20",
                "hadm_id": 20,
                "stay_id": None,
                "encounter_source": "hospital_only",
                "linked_ed_stay_flag": 0,
                "linked_hospital_admission_flag": 1,
                "admission_type": "ELECTIVE",
                "admittime": "2125-03-19 20:00:00",
                "dischtime": "2125-03-20 09:00:00",
                "hospital_length_of_stay_days": 0.542,
                "intime": None,
                "outtime": None,
                "ed_length_of_stay_hours": None,
                "ed_disposition": None,
                "arrival_transport": None,
                "encounter_start": "2125-03-19 20:00:00",
                "encounter_end": "2125-03-20 09:00:00",
                "source_tables_json": "[\"patients\", \"admissions\"]",
                "source_record_provenance_json": "{\"patients\": {\"role\": \"patients\"}, \"admissions\": {\"role\": \"admissions\"}, \"edstays\": null}",
                "encounter_index_build_run_id": "encounter-index-test",
                "encounter_index_contract_version": ENCOUNTER_INDEX_CONTRACT_VERSION,
            },
        ]
    )


def _medication_events_for_builder_test() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _medication_event_row(
                subject_id=1,
                hadm_id=10,
                encounter_id="hadm:10",
                medication_event_id="order-furosemide-10",
                raw_medication_name="Furosemide 20 mg",
                medication_name="Furosemide",
                medication_normalized="furosemide",
                event_time="2125-03-19 19:00:00",
                starttime="2125-03-19 19:00:00",
                stoptime="2125-03-20 12:00:00",
                route="PO",
                frequency="BID",
                status="active",
            ),
            _medication_event_row(
                subject_id=2,
                hadm_id=20,
                encounter_id="hadm:20",
                medication_event_id="order-lisinopril-20",
                raw_medication_name="Lisinopril 10 mg",
                medication_name="Lisinopril",
                medication_normalized="lisinopril",
                event_time="2125-03-19 21:00:00",
                starttime="2125-03-19 21:00:00",
                stoptime="2125-03-20 08:00:00",
                route="PO",
                frequency="DAILY",
                status="active",
            ),
        ]
    )


def _medication_event_row(
    *,
    subject_id: int,
    hadm_id: int,
    encounter_id: str,
    medication_event_id: str,
    raw_medication_name: str,
    medication_name: str,
    medication_normalized: str,
    event_time: str,
    starttime: str,
    stoptime: str,
    route: str,
    frequency: str,
    status: str,
) -> dict[str, object]:
    return {
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "stay_id": None,
        "encounter_id": encounter_id,
        "encounter_source": "hospital_only",
        "encounter_start": "2125-03-19 18:00:00",
        "encounter_end": "2125-03-20 10:00:00",
        "medication_event_id": medication_event_id,
        "medication_event_type": "hospital_order",
        "event_source_category": "hospital_medication_order",
        "event_source_table": "prescriptions",
        "raw_medication_name": raw_medication_name,
        "medication_name": medication_name,
        "medication_normalized": medication_normalized,
        "medication_prestandardized_text": medication_normalized,
        "event_time": event_time,
        "starttime": starttime,
        "stoptime": stoptime,
        "route": route,
        "frequency": frequency,
        "status": status,
        "dose_value": "10",
        "dose_unit": "mg",
        "pharmacy_id": 500,
        "poe_id": 700,
        "emar_id": None,
        "emar_seq": None,
        "source_home_medrecon": 0,
        "source_ed_pyxis": 0,
        "source_hospital_order": 1,
        "source_hospital_admin": 0,
        "pharmacy_enriched_flag": 1,
        "order_enrichment_applied_flag": 1,
        "order_enrichment_source_table": "pharmacy",
        "continued_from_home_inferred": 0,
        "continued_from_home_inferred_flag": 0,
        "newly_started_during_encounter_inferred": 1,
        "newly_started_during_encounter_inferred_flag": 1,
        "continuity_inference_rule": "test_rule",
        "medication_episode_id": f"episode-{medication_event_id}",
        "prescription_segment_count": 1,
        "prescription_segments_json": "[]",
        "source_tables_json": "[\"prescriptions\"]",
        "source_record_provenance_json": "{\"role\": \"prescriptions\"}",
        "medication_event_build_run_id": "medication-events-test",
        "medication_event_contract_version": MEDICATION_EVENTS_CONTRACT_VERSION,
    }


if __name__ == "__main__":
    unittest.main()
