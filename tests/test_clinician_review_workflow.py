"""Focused tests for the Phase 5 clinician review workflow artifacts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from opti_med.api.clinician_reviews import (
    ClinicianReviewRepository,
    build_review_queue_hints,
)
from opti_med.config import Settings


class ClinicianReviewWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.settings = Settings(
            analytical_root=self.root / "data" / "analytical",
            modeling_root=self.root / "data" / "modeling",
            label_root=self.root / "data" / "labels",
            default_clinician_reviewer_id="unit_test_pharmacist",
        )
        self.settings.analytical_root.mkdir(parents=True, exist_ok=True)
        self.settings.modeling_root.mkdir(parents=True, exist_ok=True)
        self.settings.label_root.mkdir(parents=True, exist_ok=True)
        self._write_reviewable_inputs()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_save_review_materializes_append_only_snapshot_and_qc(self) -> None:
        repository = ClinicianReviewRepository(self.settings)

        saved_review, first_report = repository.save_review(
            submission={
                "subject_id": 1001,
                "encounter_id": "hadm:2001",
                "medication_standardized": "olanzapine",
                "review_timestamp": "2125-03-20 10:00:00",
                "label__clinician_priority_level": "high",
                "label__clinician_priority_score": 7,
                "label__clinician_review_status": "reviewed",
                "label__clinician_reason_tags": ["polypharmacy", "renal_risk"],
                "label__clinician_note": "Needs pharmacist follow-up.",
                "label__clinician_suggested_action": "monitor",
            }
        )

        self.assertEqual(saved_review["review_version"], 1)
        self.assertEqual(saved_review["reviewer_id"], "unit_test_pharmacist")
        self.assertEqual(saved_review["modeling__row_id"], self._expected_row_id("olanzapine"))
        self.assertEqual(first_report["clinician_reviewed_row_count"], 1)
        self.assertEqual(first_report["numeric_score_populated_count"], 1)
        self.assertIn("label collection milestone", first_report["phase_scope_statement"])
        self.assertEqual(first_report["priority_level_frequencies"]["high"], 1)

        latest_snapshot = repository.load_latest_reviews()
        self.assertEqual(len(latest_snapshot), 1)
        self.assertEqual(
            latest_snapshot.iloc[0]["label__clinician_reason_tags"],
            ["polypharmacy", "renal_risk"],
        )

        updated_review, second_report = repository.save_review(
            submission={
                "subject_id": 1001,
                "encounter_id": "hadm:2001",
                "medication_standardized": "olanzapine",
                "review_timestamp": "2125-03-20 10:00:00",
                "label__clinician_priority_level": "medium",
                "label__clinician_review_status": "uncertain",
                "label__clinician_reason_tags": ["other"],
                "label__clinician_note": "Still discussing with the team.",
            }
        )

        self.assertEqual(updated_review["review_version"], 2)
        self.assertEqual(second_report["clinician_reviewed_row_count"], 1)
        self.assertEqual(second_report["required_level_populated_count"], 1)
        self.assertEqual(second_report["numeric_score_populated_count"], 0)
        self.assertEqual(second_report["reason_tag_row_coverage_count"], 1)
        self.assertEqual(
            second_report["traceability_validation"]["matched_to_modeling_row_id_count"],
            1,
        )
        self.assertEqual(
            second_report["traceability_validation"]["reviewable_duplicate_key_count"],
            0,
        )

        events = repository.load_review_events()
        self.assertEqual(len(events), 2)
        latest_snapshot = repository.load_latest_reviews()
        self.assertEqual(len(latest_snapshot), 1)
        self.assertEqual(latest_snapshot.iloc[0]["review_version"], 2)
        self.assertEqual(
            latest_snapshot.iloc[0]["label__clinician_review_status"],
            "uncertain",
        )
        self.assertTrue(self.settings.clinician_review_event_log_path.exists())
        self.assertTrue(self.settings.clinician_review_snapshot_output_path.exists())
        self.assertTrue(self.settings.clinician_review_qc_summary_path.exists())
        self.assertTrue(self.settings.clinician_review_qc_report_path.exists())

    def test_resolve_reviewable_row_and_queue_hints_use_analytical_grain(self) -> None:
        repository = ClinicianReviewRepository(self.settings)

        reviewable_row = repository.resolve_reviewable_row(
            subject_id=1001,
            encounter_id="hadm:2001",
            review_timestamp="2125-03-20 10:00:00",
            medication_candidates=["OLANZAPINE", "Olanzapine"],
            selected_event_id="hospital-order-1001-2001-1",
        )

        self.assertIsNotNone(reviewable_row)
        assert reviewable_row is not None
        self.assertEqual(reviewable_row["medication_standardized"], "olanzapine")
        self.assertEqual(reviewable_row["modeling__row_id"], self._expected_row_id("olanzapine"))

        priority, reasons = build_review_queue_hints(
            reviewable_row=reviewable_row,
            clinician_review=None,
            displayed_rule_level="high",
        )
        self.assertEqual(priority, "priority")
        self.assertIn("lacks_clinician_review", reasons)
        self.assertIn("rule_signal_present", reasons)

        disagreement_priority, disagreement_reasons = build_review_queue_hints(
            reviewable_row=reviewable_row,
            clinician_review={"label__clinician_priority_level": "low"},
            displayed_rule_level="high",
        )
        self.assertEqual(disagreement_priority, "disagreement_candidate")
        self.assertIn("clinician_rule_disagreement", disagreement_reasons)

    def test_save_review_rejects_modeling_row_id_mismatch(self) -> None:
        repository = ClinicianReviewRepository(self.settings)

        with self.assertRaisesRegex(ValueError, "modeling__row_id"):
            repository.save_review(
                submission={
                    "subject_id": 1001,
                    "encounter_id": "hadm:2001",
                    "medication_standardized": "olanzapine",
                    "review_timestamp": "2125-03-20 10:00:00",
                    "modeling__row_id": "wrong-row-id",
                    "label__clinician_priority_level": "high",
                    "label__clinician_review_status": "reviewed",
                }
            )

    def test_resolve_reviewable_row_refuses_ambiguous_text_only_matches(self) -> None:
        repository = ClinicianReviewRepository(self.settings)
        ambiguous_row = pd.DataFrame(
            [
                {
                    "subject_id": 1001,
                    "encounter_id": "hadm:2001",
                    "hadm_id": 2001,
                    "stay_id": 3001,
                    "review_timestamp": "2125-03-20 10:00:00",
                    "medication_standardized": "olanzapine_alt",
                    "selected_medication_event_id": "hospital-order-1001-2001-2",
                    "selected_medication_event_type": "hospital_order",
                    "medication_normalized": "olanzapine",
                    "medication_class_standardized": "antipsychotic",
                    "first_scope_supported_class_flag": 1,
                    "medication_status_at_review": "active_at_review_time",
                    "active_at_review_flag": 1,
                    "dose_value": "5",
                    "dose_unit": "mg",
                    "route": "PO",
                    "frequency": "HS",
                    "encounter_medication_first_scope_contract_version": "encounter_medication_first_scope_65plus.v1",
                }
            ]
        )
        current = pd.read_parquet(self.settings.encounter_medication_first_scope_output_path)
        pd.concat([current, ambiguous_row], ignore_index=True).to_parquet(
            self.settings.encounter_medication_first_scope_output_path,
            index=False,
        )

        resolved = repository.resolve_reviewable_row(
            subject_id=1001,
            encounter_id="hadm:2001",
            review_timestamp="2125-03-20 10:00:00",
            medication_candidates=["olanzapine"],
            selected_event_id=None,
        )
        self.assertIsNone(resolved)

    def _write_reviewable_inputs(self) -> None:
        first_scope = pd.DataFrame(
            [
                {
                    "subject_id": 1001,
                    "encounter_id": "hadm:2001",
                    "hadm_id": 2001,
                    "stay_id": 3001,
                    "review_timestamp": "2125-03-20 10:00:00",
                    "medication_standardized": "olanzapine",
                    "selected_medication_event_id": "hospital-order-1001-2001-1",
                    "selected_medication_event_type": "hospital_order",
                    "medication_normalized": "olanzapine",
                    "medication_class_standardized": "antipsychotic",
                    "first_scope_supported_class_flag": 1,
                    "medication_status_at_review": "active_at_review_time",
                    "active_at_review_flag": 1,
                    "dose_value": "2.5",
                    "dose_unit": "mg",
                    "route": "PO",
                    "frequency": "HS",
                    "encounter_medication_first_scope_contract_version": "encounter_medication_first_scope_65plus.v1",
                },
                {
                    "subject_id": 1002,
                    "encounter_id": "hadm:2002",
                    "hadm_id": 2002,
                    "stay_id": 3002,
                    "review_timestamp": "2125-03-21 11:00:00",
                    "medication_standardized": "omeprazole",
                    "selected_medication_event_id": "hospital-order-1002-2002-1",
                    "selected_medication_event_type": "hospital_order",
                    "medication_normalized": "omeprazole",
                    "medication_class_standardized": "ppi",
                    "first_scope_supported_class_flag": 1,
                    "medication_status_at_review": "active_at_review_time",
                    "active_at_review_flag": 1,
                    "dose_value": "20",
                    "dose_unit": "mg",
                    "route": "PO",
                    "frequency": "daily",
                    "encounter_medication_first_scope_contract_version": "encounter_medication_first_scope_65plus.v1",
                },
            ]
        )
        first_scope.to_parquet(self.settings.encounter_medication_first_scope_output_path, index=False)

        dataset = pd.DataFrame(
            [
                {
                    "subject_id": 1001,
                    "encounter_id": "hadm:2001",
                    "medication_standardized": "olanzapine",
                    "review_timestamp": "2125-03-20 10:00:00",
                    "meta__dataset_contract_version": "encounter_medication_dataset_v1_65plus_first_scope.v1",
                    "benchmark__current_rule_score": 8.0,
                    "benchmark__current_rule_available_flag": 1,
                    "benchmark__medication_class_only_medication_class_standardized": "antipsychotic",
                    "benchmark__heuristic_any_supported_class_flag": 1,
                    "label__primary_action_label": "action_undetermined",
                    "label__unknown_or_insufficient_evidence_flag": 1,
                    "meta__dataset_row_eligible_for_training_flag": 0,
                }
            ]
        )
        dataset.to_parquet(self.settings.first_scope_dataset_output_path, index=False)

    def _expected_row_id(self, medication_standardized: str) -> str:
        return (
            "subject_id=1001|encounter_id=hadm:2001|"
            f"medication_standardized={medication_standardized}|"
            "review_timestamp=2125-03-20 10:00:00"
        )


if __name__ == "__main__":
    unittest.main()
