"""Focused tests for post-blind clinician vs model comparison artifacts."""

from __future__ import annotations

import unittest

import pandas as pd

from opti_med.modeling.first_scope_post_blind_eval_comparison import (
    build_first_scope_post_blind_eval_comparison,
)


class FirstScopePostBlindEvalComparisonTests(unittest.TestCase):
    def test_builder_reports_level_agreement_and_one_vs_rest_confusion(self) -> None:
        blind_slice = pd.DataFrame(
            [
                {
                    "modeling__row_id": "row-1",
                    "subject_id": 1,
                    "encounter_id": "enc-1",
                    "medication_standardized": "drug-a",
                    "review_timestamp": "2026-01-01 00:00:00",
                    "medication_class_standardized": "opioid",
                    "queue_rank": 1,
                },
                {
                    "modeling__row_id": "row-2",
                    "subject_id": 2,
                    "encounter_id": "enc-2",
                    "medication_standardized": "drug-b",
                    "review_timestamp": "2026-01-02 00:00:00",
                    "medication_class_standardized": "ppi",
                    "queue_rank": 2,
                },
                {
                    "modeling__row_id": "row-3",
                    "subject_id": 3,
                    "encounter_id": "enc-3",
                    "medication_standardized": "drug-c",
                    "review_timestamp": "2026-01-03 00:00:00",
                    "medication_class_standardized": "benzodiazepine",
                    "queue_rank": 3,
                },
                {
                    "modeling__row_id": "row-4",
                    "subject_id": 4,
                    "encounter_id": "enc-4",
                    "medication_standardized": "drug-d",
                    "review_timestamp": "2026-01-04 00:00:00",
                    "medication_class_standardized": "antipsychotic",
                    "queue_rank": 4,
                },
            ]
        )
        latest_reviews = pd.DataFrame(
            [
                {
                    "modeling__row_id": "row-1",
                    "subject_id": 1,
                    "encounter_id": "enc-1",
                    "medication_standardized": "drug-a",
                    "review_timestamp": "2026-01-01 00:00:00",
                    "benchmark__medication_class_only_medication_class_standardized": "opioid",
                    "review_submission_id": "sub-1",
                    "review_submission_timestamp": "2026-03-25T12:00:00Z",
                    "review_version": 1,
                    "reviewer_id": "pharmacist",
                    "label__clinician_priority_level": "low",
                    "label__clinician_review_status": "reviewed",
                    "review_provenance_json": {
                        "submission_source": "blind_eval_slice_ui_phase8"
                    },
                },
                {
                    "modeling__row_id": "row-2",
                    "subject_id": 2,
                    "encounter_id": "enc-2",
                    "medication_standardized": "drug-b",
                    "review_timestamp": "2026-01-02 00:00:00",
                    "benchmark__medication_class_only_medication_class_standardized": "ppi",
                    "review_submission_id": "sub-2",
                    "review_submission_timestamp": "2026-03-25T12:01:00Z",
                    "review_version": 1,
                    "reviewer_id": "pharmacist",
                    "label__clinician_priority_level": "medium",
                    "label__clinician_review_status": "reviewed",
                    "review_provenance_json": {
                        "submission_source": "blind_eval_slice_ui_phase8"
                    },
                },
                {
                    "modeling__row_id": "row-3",
                    "subject_id": 3,
                    "encounter_id": "enc-3",
                    "medication_standardized": "drug-c",
                    "review_timestamp": "2026-01-03 00:00:00",
                    "benchmark__medication_class_only_medication_class_standardized": "benzodiazepine",
                    "review_submission_id": "sub-3",
                    "review_submission_timestamp": "2026-03-25T12:02:00Z",
                    "review_version": 1,
                    "reviewer_id": "pharmacist",
                    "label__clinician_priority_level": "high",
                    "label__clinician_review_status": "reviewed",
                    "review_provenance_json": {
                        "submission_source": "blind_eval_slice_ui_phase8"
                    },
                },
                {
                    "modeling__row_id": "row-4",
                    "subject_id": 4,
                    "encounter_id": "enc-4",
                    "medication_standardized": "drug-d",
                    "review_timestamp": "2026-01-04 00:00:00",
                    "benchmark__medication_class_only_medication_class_standardized": "antipsychotic",
                    "review_submission_id": "sub-4",
                    "review_submission_timestamp": "2026-03-25T12:03:00Z",
                    "review_version": 1,
                    "reviewer_id": "pharmacist",
                    "label__clinician_priority_level": "high",
                    "label__clinician_review_status": "reviewed",
                    "review_provenance_json": {
                        "submission_source": "blind_eval_slice_ui_phase8"
                    },
                },
            ]
        )
        scored_universe = pd.DataFrame(
            [
                {
                    "modeling__row_id": "row-1",
                    "prediction__ordinal_priority_level": "low",
                    "prediction__probability_low": 0.8,
                    "prediction__probability_medium": 0.1,
                    "prediction__probability_high": 0.1,
                    "prediction__medium_high_margin": 0.0,
                    "benchmark__current_rule_score_level": pd.NA,
                },
                {
                    "modeling__row_id": "row-2",
                    "prediction__ordinal_priority_level": "high",
                    "prediction__probability_low": 0.1,
                    "prediction__probability_medium": 0.3,
                    "prediction__probability_high": 0.6,
                    "prediction__medium_high_margin": 0.3,
                    "benchmark__current_rule_score_level": pd.NA,
                },
                {
                    "modeling__row_id": "row-3",
                    "prediction__ordinal_priority_level": "high",
                    "prediction__probability_low": 0.1,
                    "prediction__probability_medium": 0.2,
                    "prediction__probability_high": 0.7,
                    "prediction__medium_high_margin": 0.5,
                    "benchmark__current_rule_score_level": pd.NA,
                },
                {
                    "modeling__row_id": "row-4",
                    "prediction__ordinal_priority_level": "medium",
                    "prediction__probability_low": 0.1,
                    "prediction__probability_medium": 0.5,
                    "prediction__probability_high": 0.4,
                    "prediction__medium_high_margin": 0.1,
                    "benchmark__current_rule_score_level": pd.NA,
                },
            ]
        )

        result = build_first_scope_post_blind_eval_comparison(
            blind_eval_slice=blind_slice,
            latest_clinician_reviews=latest_reviews,
            scored_universe=scored_universe,
        )

        self.assertEqual(result.summary["candidate_row_count"], 4)
        self.assertEqual(result.summary["blind_session_latest_review_row_count"], 4)
        self.assertEqual(result.summary["blind_slice_overlap_row_count"], 4)
        self.assertEqual(result.summary["completed_blind_review_row_count"], 4)
        self.assertEqual(result.summary["evaluation_eligible_row_count"], 4)
        comparison = result.summary["model_vs_clinician_level_comparison"]
        self.assertEqual(comparison["same_level_agreement_count"], 2)
        self.assertAlmostEqual(comparison["same_level_agreement_rate"], 0.5, places=6)
        self.assertEqual(comparison["off_by_one_level_count"], 2)
        self.assertEqual(
            comparison["ordinal_confusion_matrix"],
            {
                "low": {"low": 1, "medium": 0, "high": 0},
                "medium": {"low": 0, "medium": 0, "high": 1},
                "high": {"low": 0, "medium": 1, "high": 1},
            },
        )
        high_confusion = result.summary["one_vs_rest_confusion_by_level"]["high"]
        self.assertEqual(high_confusion["true_positive_count"], 1)
        self.assertEqual(high_confusion["true_negative_count"], 1)
        self.assertEqual(high_confusion["false_positive_count"], 1)
        self.assertEqual(high_confusion["false_negative_count"], 1)
        self.assertIn("same-level agreement", result.report_markdown.lower())


if __name__ == "__main__":
    unittest.main()
