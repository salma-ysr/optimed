"""Focused tests for the targeted blind clinician evaluation slice."""

from __future__ import annotations

import unittest

import pandas as pd

from opti_med.api.clinician_reviews import (
    REVIEWABLE_COLUMNS,
    overlay_latest_reviews_on_queue_rows,
)
from opti_med.modeling.first_scope_targeted_blind_eval import (
    build_first_scope_targeted_blind_eval,
    filter_targeted_blind_eval_slice,
)
from tests.test_first_scope_dataset_v1 import (
    build_synthetic_dataset_with_rule_benchmark,
    expand_dataset_for_split_tests,
)
from tests.test_first_scope_ordinal_baseline_modeling import (
    _make_manual_splits,
    _make_ordinal_targets,
)
from opti_med.modeling.first_scope_ordinal_targets import build_first_scope_ordinal_targets


class FirstScopeTargetedBlindEvalTests(unittest.TestCase):
    def test_builder_surfaces_uncertain_and_boundary_candidates(self) -> None:
        dataset = expand_dataset_for_split_tests(
            build_synthetic_dataset_with_rule_benchmark()["dataset"]
        )
        dataset["feature__age_context"] = list(range(70, 70 + len(dataset)))
        ordinal_targets_result = build_first_scope_ordinal_targets(
            encounter_medication_dataset=dataset,
            latest_clinician_reviews=_make_ordinal_targets(dataset),
            splits=_make_manual_splits(dataset),
        )
        reviewable_universe = _build_reviewable_universe(dataset)
        latest_reviews = _make_ordinal_targets(dataset)
        latest_reviews["hadm_id"] = dataset["meta__hadm_id"].tolist()
        latest_reviews["stay_id"] = dataset["meta__stay_id"].tolist()
        latest_reviews["medication_normalized"] = dataset["medication_standardized"].tolist()
        latest_reviews["review_artifact_version"] = "synthetic-review.v1"
        latest_reviews["label__clinician_reviewed_flag"] = 1
        latest_reviews["review_provenance_json"] = [{} for _ in range(len(latest_reviews))]
        latest_reviews["label__clinician_note"] = pd.NA
        latest_reviews["label__clinician_suggested_action"] = pd.NA

        result = build_first_scope_targeted_blind_eval(
            encounter_medication_dataset=dataset,
            reviewable_universe=reviewable_universe,
            latest_clinician_reviews=latest_reviews,
            ordinal_targets=ordinal_targets_result.targets,
            random_seed=17,
            fallback_minimum_candidate_rows=1,
        )

        self.assertGreaterEqual(len(result.blind_eval_slice), 1)
        self.assertGreaterEqual(
            result.summary["uncertain_existing_review_candidate_count"],
            1,
        )
        filtered = filter_targeted_blind_eval_slice(
            result.blind_eval_slice,
            review_status="uncertain",
        )
        self.assertTrue(
            filtered["label__clinician_review_status"].fillna("").astype(str).str.lower().eq("uncertain").all()
        )
        self.assertIn("real ordinal logistic model", result.report_markdown)

    def test_overlay_latest_reviews_keeps_fixed_blind_session_order(self) -> None:
        dataset = expand_dataset_for_split_tests(
            build_synthetic_dataset_with_rule_benchmark()["dataset"]
        )
        dataset["feature__age_context"] = list(range(70, 70 + len(dataset)))
        latest_reviews = _make_ordinal_targets(dataset)
        latest_reviews["hadm_id"] = dataset["meta__hadm_id"].tolist()
        latest_reviews["stay_id"] = dataset["meta__stay_id"].tolist()
        latest_reviews["medication_normalized"] = dataset["medication_standardized"].tolist()
        latest_reviews["review_artifact_version"] = "synthetic-review.v1"
        latest_reviews["label__clinician_reviewed_flag"] = 1
        latest_reviews["review_provenance_json"] = [{} for _ in range(len(latest_reviews))]
        latest_reviews["label__clinician_note"] = pd.NA
        latest_reviews["label__clinician_suggested_action"] = pd.NA
        ordinal_targets_result = build_first_scope_ordinal_targets(
            encounter_medication_dataset=dataset,
            latest_clinician_reviews=latest_reviews,
            splits=_make_manual_splits(dataset),
        )
        reviewable_universe = _build_reviewable_universe(dataset)
        result = build_first_scope_targeted_blind_eval(
            encounter_medication_dataset=dataset,
            reviewable_universe=reviewable_universe,
            latest_clinician_reviews=latest_reviews,
            ordinal_targets=ordinal_targets_result.targets,
            random_seed=17,
            fallback_minimum_candidate_rows=1,
        )

        fixed_slice = result.blind_eval_slice.copy()
        original_row_ids = fixed_slice["modeling__row_id"].tolist()
        target_row = fixed_slice.iloc[0]
        updated_latest_reviews = latest_reviews.loc[
            ~(
                latest_reviews["subject_id"].eq(target_row["subject_id"])
                & latest_reviews["encounter_id"].eq(target_row["encounter_id"])
                & latest_reviews["medication_standardized"].eq(target_row["medication_standardized"])
                & latest_reviews["review_timestamp"].eq(target_row["review_timestamp"])
            )
        ].copy()
        updated_latest_reviews = pd.concat(
            [
                updated_latest_reviews,
                pd.DataFrame(
                    [
                        {
                            **{
                                column_name: pd.NA
                                for column_name in latest_reviews.columns
                            },
                            "subject_id": target_row["subject_id"],
                            "encounter_id": target_row["encounter_id"],
                            "hadm_id": target_row["hadm_id"],
                            "stay_id": target_row["stay_id"],
                            "medication_standardized": target_row["medication_standardized"],
                            "medication_normalized": target_row["medication_normalized"],
                            "review_timestamp": target_row["review_timestamp"],
                            "modeling__row_id": target_row["modeling__row_id"],
                            "review_submission_id": "synthetic-rereview",
                            "review_version": 99,
                            "review_artifact_version": "synthetic-review.v1",
                            "review_submission_timestamp": "2026-03-25T12:00:00Z",
                            "reviewer_id": "pharmacist_second_pass",
                            "label__clinician_priority_level": "high",
                            "label__clinician_priority_score": 9,
                            "label__clinician_priority_score_level": "high",
                            "label__clinician_review_status": "reviewed",
                            "label__clinician_reason_tags": ["monitoring_needed"],
                            "label__clinician_note": "Updated during blind session.",
                            "label__clinician_reviewed_flag": 1,
                            "label__clinician_suggested_action": "monitor",
                            "review_provenance_json": {
                                "submission_source": "blind_eval_slice_ui_phase8"
                            },
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        overlaid = overlay_latest_reviews_on_queue_rows(
            queue_rows=fixed_slice,
            latest_reviews=updated_latest_reviews,
        )

        self.assertEqual(len(overlaid), len(fixed_slice))
        self.assertEqual(overlaid["modeling__row_id"].tolist(), original_row_ids)
        refreshed_row = overlaid.loc[
            overlaid["modeling__row_id"] == target_row["modeling__row_id"]
        ].iloc[0]
        self.assertEqual(refreshed_row["review_submission_id"], "synthetic-rereview")
        self.assertEqual(refreshed_row["review_version"], 99)
        self.assertEqual(refreshed_row["label__clinician_note"], "Updated during blind session.")

    def test_fallback_candidates_are_capped_to_minimum_needed(self) -> None:
        dataset = expand_dataset_for_split_tests(
            build_synthetic_dataset_with_rule_benchmark()["dataset"]
        )
        dataset["feature__age_context"] = list(range(70, 70 + len(dataset)))
        latest_reviews = _make_ordinal_targets(dataset)
        latest_reviews["hadm_id"] = dataset["meta__hadm_id"].tolist()
        latest_reviews["stay_id"] = dataset["meta__stay_id"].tolist()
        latest_reviews["medication_normalized"] = dataset["medication_standardized"].tolist()
        latest_reviews["review_artifact_version"] = "synthetic-review.v1"
        latest_reviews["label__clinician_reviewed_flag"] = 1
        latest_reviews["review_provenance_json"] = [{} for _ in range(len(latest_reviews))]
        latest_reviews["label__clinician_note"] = pd.NA
        latest_reviews["label__clinician_suggested_action"] = pd.NA
        latest_reviews["label__clinician_review_status"] = "reviewed"
        ordinal_targets_result = build_first_scope_ordinal_targets(
            encounter_medication_dataset=dataset,
            latest_clinician_reviews=latest_reviews,
            splits=_make_manual_splits(dataset),
        )
        reviewable_universe = _build_reviewable_universe(dataset)
        reviewable_universe["benchmark__current_rule_score"] = pd.NA
        reviewable_universe["benchmark__current_rule_score_level"] = pd.NA
        reviewable_universe["benchmark__current_rule_available_flag"] = 0

        result = build_first_scope_targeted_blind_eval(
            encounter_medication_dataset=dataset,
            reviewable_universe=reviewable_universe,
            latest_clinician_reviews=latest_reviews.iloc[0:0].copy(),
            ordinal_targets=ordinal_targets_result.targets,
            random_seed=17,
            medium_high_margin_threshold=-1.0,
            fallback_high_probability_threshold=0.0,
            fallback_minimum_candidate_rows=5,
        )

        self.assertEqual(len(result.blind_eval_slice), 5)
        self.assertEqual(result.summary["high_signal_unreviewed_fallback_candidate_count"], 5)


def _build_reviewable_universe(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in dataset.to_dict(orient="records"):
        row = {column_name: pd.NA for column_name in REVIEWABLE_COLUMNS}
        row.update(
            {
                "subject_id": record["subject_id"],
                "encounter_id": record["encounter_id"],
                "hadm_id": record["meta__hadm_id"],
                "stay_id": record["meta__stay_id"],
                "review_timestamp": record["review_timestamp"],
                "medication_standardized": record["medication_standardized"],
                "selected_medication_event_id": f"synthetic-{record['subject_id']}",
                "selected_medication_event_type": "hospital_order",
                "medication_normalized": record["medication_standardized"],
                "medication_class_standardized": record[
                    "benchmark__medication_class_only_medication_class_standardized"
                ],
                "first_scope_supported_class_flag": 1,
                "medication_status_at_review": "active_at_review_time",
                "active_at_review_flag": 1,
                "dose_value": "1",
                "dose_unit": "mg",
                "route": "PO",
                "frequency": "daily",
                "encounter_medication_first_scope_contract_version": "synthetic-first-scope.v1",
                "meta__dataset_contract_version": record["meta__dataset_contract_version"],
                "benchmark__current_rule_score": record["benchmark__current_rule_score"],
                "benchmark__current_rule_score_level": record["benchmark__current_rule_score_level"],
                "benchmark__current_rule_available_flag": record["benchmark__current_rule_available_flag"],
                "benchmark__medication_class_only_medication_class_standardized": record[
                    "benchmark__medication_class_only_medication_class_standardized"
                ],
                "benchmark__heuristic_any_supported_class_flag": record[
                    "benchmark__heuristic_any_supported_class_flag"
                ],
                "label__primary_action_label": record["label__primary_action_label"],
                "label__unknown_or_insufficient_evidence_flag": record[
                    "label__unknown_or_insufficient_evidence_flag"
                ],
                "meta__dataset_row_eligible_for_training_flag": record[
                    "meta__dataset_row_eligible_for_training_flag"
                ],
                "modeling__row_id": (
                    f"subject_id={record['subject_id']}|encounter_id={record['encounter_id']}|"
                    f"medication_standardized={record['medication_standardized']}|"
                    f"review_timestamp={record['review_timestamp']}"
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
