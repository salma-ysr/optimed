"""Tests for encounter-relative medication snapshot helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from opti_med.data_access.medication_snapshot import (
    build_encounter_medication_review,
    build_medication_snapshot,
    classify_medication_status_at_review_time,
    filter_active_medication_events,
    interval_overlaps_snapshot,
    is_medication_event_active_at_snapshot,
    select_review_timestamp_metadata_for_encounter,
    select_snapshot_time_for_encounter,
)


class MedicationSnapshotHelperTests(unittest.TestCase):
    def test_interval_overlaps_snapshot_with_open_ended_interval(self) -> None:
        self.assertTrue(
            interval_overlaps_snapshot(
                "2201-10-30 12:00:00",
                None,
                "2201-10-31 12:00:00",
            )
        )

    def test_interval_overlaps_snapshot_respects_stop_time(self) -> None:
        self.assertFalse(
            interval_overlaps_snapshot(
                "2201-10-30 12:00:00",
                "2201-10-30 18:00:00",
                "2201-10-31 12:00:00",
            )
        )

    def test_select_snapshot_time_for_ed_strategy(self) -> None:
        encounter = {
            "intime": "2125-03-19 12:36:00",
            "outtime": "2125-03-19 16:59:47",
            "admittime": "2125-03-19 18:00:00",
            "dischtime": "2125-03-20 10:00:00",
            "encounter_start": "2125-03-19 12:36:00",
            "encounter_end": "2125-03-20 10:00:00",
        }
        snapshot = select_snapshot_time_for_encounter(encounter, "ed")
        self.assertEqual(str(snapshot), "2125-03-19 16:59:47")

    def test_select_snapshot_time_for_hospital_strategy(self) -> None:
        encounter = {
            "intime": "2125-03-19 12:36:00",
            "outtime": "2125-03-19 16:59:47",
            "admittime": "2125-03-19 18:00:00",
            "dischtime": "2125-03-20 10:00:00",
            "encounter_start": "2125-03-19 12:36:00",
            "encounter_end": "2125-03-20 10:00:00",
        }
        snapshot = select_snapshot_time_for_encounter(encounter, "hospital")
        self.assertEqual(str(snapshot), "2125-03-20 10:00:00")

    def test_select_review_timestamp_prefers_labs_over_encounter_end_when_no_meds(self) -> None:
        encounter = {
            "subject_id": 1,
            "hadm_id": 10,
            "stay_id": None,
            "encounter_id": "hadm:10",
            "encounter_start": "2125-03-19 12:36:00",
            "encounter_end": "2125-03-20 10:00:00",
            "dischtime": "2125-03-20 10:00:00",
        }
        labs = pd.DataFrame(
            [
                {"hadm_id": 10, "charttime": "2125-03-20 08:00:00"},
                {"hadm_id": 10, "charttime": "2125-03-20 09:00:00"},
            ]
        )
        review_time, review_source = select_review_timestamp_metadata_for_encounter(
            encounter,
            strategy="latest_available",
            medication_events=pd.DataFrame(),
            labevents=labs,
            triage=None,
            vitalsign=None,
        )
        self.assertEqual(str(review_time), "2125-03-20 09:00:00")
        self.assertEqual(review_source, "lab")

    def test_select_review_timestamp_uses_precomputed_lab_lookup(self) -> None:
        encounter = {
            "subject_id": 1,
            "hadm_id": 10,
            "stay_id": None,
            "encounter_id": "hadm:10",
            "encounter_start": "2125-03-19 12:36:00",
            "encounter_end": "2125-03-20 10:00:00",
            "dischtime": "2125-03-20 10:00:00",
        }
        review_time, review_source = select_review_timestamp_metadata_for_encounter(
            encounter,
            strategy="latest_available",
            medication_events=pd.DataFrame(),
            labevents=None,
            triage=None,
            vitalsign=None,
            latest_lab_timestamp_by_hadm_id={
                10: pd.Timestamp("2125-03-20 09:00:00"),
            },
        )
        self.assertEqual(str(review_time), "2125-03-20 09:00:00")
        self.assertEqual(review_source, "lab")

    def test_hospital_order_is_active_at_later_snapshot(self) -> None:
        row = {
            "medication_event_type": "hospital_order",
            "starttime": "2201-10-30 12:00:00",
            "stoptime": None,
            "event_time": "2201-10-30 12:00:00",
        }
        self.assertTrue(
            is_medication_event_active_at_snapshot(row, "2201-10-31 12:00:00")
        )

    def test_point_event_is_only_active_at_exact_snapshot(self) -> None:
        row = {
            "medication_event_type": "hospital_admin",
            "starttime": "2201-10-30 12:00:00",
            "stoptime": None,
            "event_time": "2201-10-30 12:00:00",
        }
        self.assertFalse(
            is_medication_event_active_at_snapshot(row, "2201-10-31 12:00:00")
        )
        self.assertTrue(
            is_medication_event_active_at_snapshot(row, "2201-10-30 12:00:00")
        )

    def test_filter_active_medication_events(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "medication_event_type": "hospital_order",
                    "starttime": "2201-10-30 12:00:00",
                    "stoptime": None,
                    "event_time": "2201-10-30 12:00:00",
                    "medication_normalized": "furosemide",
                },
                {
                    "medication_event_type": "hospital_admin",
                    "starttime": "2201-10-29 08:00:00",
                    "stoptime": None,
                    "event_time": "2201-10-29 08:00:00",
                    "medication_normalized": "midazolam",
                },
            ]
        )
        active = filter_active_medication_events(events, "2201-10-31 12:00:00")
        self.assertEqual(active["medication_normalized"].tolist(), ["furosemide"])

    def test_home_medrecon_without_continuation_is_not_active(self) -> None:
        row = {
            "medication_event_type": "home_medrecon",
            "event_time": "2201-10-30 08:00:00",
            "starttime": None,
            "stoptime": None,
            "continued_from_home_inferred": 0,
        }
        self.assertFalse(
            is_medication_event_active_at_snapshot(row, "2201-10-31 12:00:00")
        )

    def test_classify_pre_admission_only_status(self) -> None:
        group = pd.DataFrame(
            [
                {
                    "medication_event_type": "home_medrecon",
                    "event_time": "2201-10-30 08:00:00",
                    "starttime": None,
                    "stoptime": None,
                    "source_home_medrecon": 1,
                    "continued_from_home_inferred": 0,
                }
            ]
        )
        self.assertEqual(
            classify_medication_status_at_review_time(group, "2201-10-31 12:00:00"),
            "pre_admission_only",
        )

    def test_build_medication_snapshot_deduplicates_same_normalized_medication(self) -> None:
        medication_events = pd.DataFrame(
            [
                {
                    "subject_id": 1,
                    "hadm_id": 10,
                    "stay_id": None,
                    "encounter_id": "hadm:10",
                    "encounter_source": "hospital_only",
                    "encounter_start": "2201-10-30 10:00:00",
                    "encounter_end": "2201-10-31 10:00:00",
                    "medication_event_id": "order-1",
                    "medication_event_type": "hospital_order",
                    "raw_medication_name": "Furosemide",
                    "medication_name": "Furosemide",
                    "medication_normalized": "furosemide",
                    "event_time": "2201-10-30 10:00:00",
                    "starttime": "2201-10-30 10:00:00",
                    "stoptime": None,
                    "route": "PO",
                    "frequency": "BID",
                    "status": None,
                    "dose_value": "20",
                    "dose_unit": "mg",
                    "source_home_medrecon": 0,
                    "source_ed_pyxis": 0,
                    "source_hospital_order": 1,
                    "source_hospital_admin": 0,
                    "pharmacy_enriched_flag": 1,
                    "continued_from_home_inferred": 0,
                    "newly_started_during_encounter_inferred": 1,
                },
                {
                    "subject_id": 1,
                    "hadm_id": 10,
                    "stay_id": None,
                    "encounter_id": "hadm:10",
                    "encounter_source": "hospital_only",
                    "encounter_start": "2201-10-30 10:00:00",
                    "encounter_end": "2201-10-31 10:00:00",
                    "medication_event_id": "home-1",
                    "medication_event_type": "home_medrecon",
                    "raw_medication_name": "Furosemide",
                    "medication_name": "Furosemide",
                    "medication_normalized": "furosemide",
                    "event_time": "2201-10-30 09:00:00",
                    "starttime": None,
                    "stoptime": None,
                    "route": None,
                    "frequency": None,
                    "status": None,
                    "dose_value": None,
                    "dose_unit": None,
                    "source_home_medrecon": 1,
                    "source_ed_pyxis": 0,
                    "source_hospital_order": 0,
                    "source_hospital_admin": 0,
                    "pharmacy_enriched_flag": 0,
                    "continued_from_home_inferred": 0,
                    "newly_started_during_encounter_inferred": 0,
                },
            ]
        )
        encounter_index = pd.DataFrame(
            [
                {
                    "subject_id": 1,
                    "hadm_id": 10,
                    "stay_id": None,
                    "encounter_id": "hadm:10",
                    "encounter_source": "hospital_only",
                    "intime": None,
                    "outtime": None,
                    "admittime": "2201-10-30 10:00:00",
                    "dischtime": "2201-10-31 10:00:00",
                    "encounter_start": "2201-10-30 10:00:00",
                    "encounter_end": "2201-10-31 10:00:00",
                }
            ]
        )
        snapshot = build_medication_snapshot(
            medication_events=medication_events,
            encounter_index=encounter_index,
            snapshot_strategy="hospital",
        )
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(snapshot.iloc[0]["selected_event_type"], "hospital_order")
        self.assertEqual(snapshot.iloc[0]["active_event_count"], 1)
        self.assertEqual(snapshot.iloc[0]["source_home_medrecon"], 1)
        self.assertEqual(snapshot.iloc[0]["source_hospital_order"], 1)

    def test_build_encounter_medication_review_preserves_inactive_history(self) -> None:
        medication_events = pd.DataFrame(
            [
                {
                    "subject_id": 1,
                    "hadm_id": 10,
                    "stay_id": None,
                    "encounter_id": "hadm:10",
                    "encounter_source": "hospital_only",
                    "encounter_start": "2201-10-30 10:00:00",
                    "encounter_end": "2201-10-31 10:00:00",
                    "medication_event_id": "order-1",
                    "medication_event_type": "hospital_order",
                    "raw_medication_name": "Furosemide",
                    "medication_name": "Furosemide",
                    "medication_normalized": "furosemide",
                    "event_time": "2201-10-30 10:00:00",
                    "starttime": "2201-10-30 10:00:00",
                    "stoptime": None,
                    "route": "PO",
                    "frequency": "BID",
                    "status": None,
                    "dose_value": "20",
                    "dose_unit": "mg",
                    "source_home_medrecon": 0,
                    "source_ed_pyxis": 0,
                    "source_hospital_order": 1,
                    "source_hospital_admin": 0,
                    "pharmacy_enriched_flag": 1,
                    "continued_from_home_inferred": 0,
                    "newly_started_during_encounter_inferred": 1,
                },
                {
                    "subject_id": 1,
                    "hadm_id": 10,
                    "stay_id": None,
                    "encounter_id": "hadm:10",
                    "encounter_source": "hospital_only",
                    "encounter_start": "2201-10-30 10:00:00",
                    "encounter_end": "2201-10-31 10:00:00",
                    "medication_event_id": "order-2",
                    "medication_event_type": "hospital_order",
                    "raw_medication_name": "Midazolam",
                    "medication_name": "Midazolam",
                    "medication_normalized": "midazolam",
                    "event_time": "2201-10-30 10:00:00",
                    "starttime": "2201-10-30 10:00:00",
                    "stoptime": "2201-10-30 12:00:00",
                    "route": "IV",
                    "frequency": "ONCE",
                    "status": None,
                    "dose_value": "2",
                    "dose_unit": "mg",
                    "source_home_medrecon": 0,
                    "source_ed_pyxis": 0,
                    "source_hospital_order": 1,
                    "source_hospital_admin": 0,
                    "pharmacy_enriched_flag": 1,
                    "continued_from_home_inferred": 0,
                    "newly_started_during_encounter_inferred": 1,
                },
            ]
        )
        encounter_index = pd.DataFrame(
            [
                {
                    "subject_id": 1,
                    "hadm_id": 10,
                    "stay_id": None,
                    "encounter_id": "hadm:10",
                    "encounter_source": "hospital_only",
                    "intime": None,
                    "outtime": None,
                    "admittime": "2201-10-30 10:00:00",
                    "dischtime": "2201-10-31 10:00:00",
                    "encounter_start": "2201-10-30 10:00:00",
                    "encounter_end": "2201-10-31 10:00:00",
                }
            ]
        )
        review = build_encounter_medication_review(
            medication_events=medication_events,
            encounter_index=encounter_index,
            snapshot_strategy="hospital",
        )
        statuses = {
            row["medication_normalized"]: row["medication_status"]
            for row in review.to_dict(orient="records")
        }
        self.assertEqual(statuses["furosemide"], "active_at_review_time")
        self.assertEqual(statuses["midazolam"], "inactive_before_review_time")


if __name__ == "__main__":
    unittest.main()
