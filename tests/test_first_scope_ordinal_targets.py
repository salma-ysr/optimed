"""Focused tests for ordinal target preparation from clinician reviews."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.evaluation.contracts import TrainValidationTestSplitConfig
from opti_med.modeling.first_scope_dataset import build_first_scope_splits
from opti_med.modeling.first_scope_ordinal_targets import (
    FIRST_SCOPE_ORDINAL_TARGET_ARTIFACT_VERSION,
    ORDINAL_TARGET_DEFAULT_EXCLUSION_REASON_COLUMN,
    ORDINAL_TARGET_DEFAULT_INCLUDE_COLUMN,
    ORDINAL_TARGET_LEVEL_COLUMN,
    attach_first_scope_ordinal_targets,
    build_first_scope_ordinal_targets,
    write_first_scope_ordinal_target_artifacts,
)
from tests.test_first_scope_dataset_v1 import (
    build_synthetic_dataset_with_rule_benchmark,
    expand_dataset_for_split_tests,
)


class FirstScopeOrdinalTargetTests(unittest.TestCase):
    def test_ordinal_target_builder_prepares_sidecar_and_default_inclusion_flags(self) -> None:
        dataset = expand_dataset_for_split_tests(
            build_synthetic_dataset_with_rule_benchmark()["dataset"]
        )
        splits = build_first_scope_splits(
            encounter_medication_dataset=dataset,
            split_config=TrainValidationTestSplitConfig(
                train_fraction=0.50,
                validation_fraction=0.25,
                test_fraction=0.25,
                random_seed=17,
                stratification_label_name="label__primary_action_binary",
            ),
        )
        clinician_reviews = pd.DataFrame(
            [
                {
                    "modeling__row_id": self._row_id(dataset.iloc[0]),
                    "subject_id": dataset.iloc[0]["subject_id"],
                    "encounter_id": dataset.iloc[0]["encounter_id"],
                    "medication_standardized": dataset.iloc[0]["medication_standardized"],
                    "review_timestamp": dataset.iloc[0]["review_timestamp"],
                    "review_submission_id": "review-1",
                    "review_submission_timestamp": "2125-03-21T11:00:00Z",
                    "review_version": 1,
                    "reviewer_id": "pharm-a",
                    "label__clinician_priority_level": "high",
                    "label__clinician_priority_score": 8.0,
                    "label__clinician_priority_score_level": "high",
                    "label__clinician_review_status": "reviewed",
                    "label__clinician_reason_tags": ["polypharmacy"],
                    "label__clinician_note": "High priority review.",
                    "label__clinician_suggested_action": "monitor",
                },
                {
                    "modeling__row_id": self._row_id(dataset.iloc[1]),
                    "subject_id": dataset.iloc[1]["subject_id"],
                    "encounter_id": dataset.iloc[1]["encounter_id"],
                    "medication_standardized": dataset.iloc[1]["medication_standardized"],
                    "review_timestamp": dataset.iloc[1]["review_timestamp"],
                    "review_submission_id": "review-2",
                    "review_submission_timestamp": "2125-03-21T11:05:00Z",
                    "review_version": 1,
                    "reviewer_id": "pharm-a",
                    "label__clinician_priority_level": "medium",
                    "label__clinician_priority_score": None,
                    "label__clinician_priority_score_level": None,
                    "label__clinician_review_status": "uncertain",
                    "label__clinician_reason_tags": [],
                    "label__clinician_note": None,
                    "label__clinician_suggested_action": None,
                },
            ]
        )

        result = build_first_scope_ordinal_targets(
            encounter_medication_dataset=dataset,
            latest_clinician_reviews=clinician_reviews,
            splits=splits,
        )

        self.assertEqual(len(result.targets), 2)
        self.assertEqual(
            result.targets.iloc[0]["target__artifact_contract_version"],
            FIRST_SCOPE_ORDINAL_TARGET_ARTIFACT_VERSION,
        )
        included = result.targets.loc[result.targets[ORDINAL_TARGET_DEFAULT_INCLUDE_COLUMN] == 1]
        excluded = result.targets.loc[result.targets[ORDINAL_TARGET_DEFAULT_INCLUDE_COLUMN] == 0]
        self.assertEqual(len(included), 1)
        self.assertEqual(included.iloc[0][ORDINAL_TARGET_LEVEL_COLUMN], "high")
        self.assertEqual(int(included.iloc[0]["target__score_level_alignment_flag"]), 1)
        self.assertEqual(len(excluded), 1)
        self.assertEqual(
            excluded.iloc[0][ORDINAL_TARGET_DEFAULT_EXCLUSION_REASON_COLUMN],
            "review_status_uncertain",
        )
        self.assertEqual(result.summary["prepared_target_row_count"], 2)
        self.assertEqual(result.summary["default_training_included_row_count"], 1)
        self.assertEqual(result.summary["review_status_frequencies"]["reviewed"], 1)
        self.assertEqual(result.summary["review_status_frequencies"]["uncertain"], 1)
        self.assertEqual(result.summary["priority_level_frequencies_all"]["high"], 1)
        self.assertEqual(result.summary["priority_level_frequencies_all"]["medium"], 1)
        self.assertIn("target-preparation milestone", result.report_markdown)

    def test_attach_ordinal_targets_merges_sidecar_back_to_dataset(self) -> None:
        dataset = expand_dataset_for_split_tests(
            build_synthetic_dataset_with_rule_benchmark()["dataset"]
        )
        review_row = {
            "modeling__row_id": self._row_id(dataset.iloc[0]),
            "subject_id": dataset.iloc[0]["subject_id"],
            "encounter_id": dataset.iloc[0]["encounter_id"],
            "medication_standardized": dataset.iloc[0]["medication_standardized"],
            "review_timestamp": dataset.iloc[0]["review_timestamp"],
            "review_submission_id": "review-1",
            "review_submission_timestamp": "2125-03-21T11:00:00Z",
            "review_version": 1,
            "reviewer_id": "pharm-a",
            "label__clinician_priority_level": "low",
            "label__clinician_priority_score": 2.0,
            "label__clinician_priority_score_level": "low",
            "label__clinician_review_status": "reviewed",
        }
        result = build_first_scope_ordinal_targets(
            encounter_medication_dataset=dataset,
            latest_clinician_reviews=pd.DataFrame([review_row]),
            splits=None,
        )

        joined = attach_first_scope_ordinal_targets(
            encounter_medication_dataset=dataset,
            ordinal_targets=result.targets,
        )

        self.assertEqual(len(joined), len(dataset))
        matched = joined.loc[joined["target__ordinal_priority_level"].notna()].copy()
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched.iloc[0]["target__ordinal_priority_level"], "low")
        self.assertEqual(int(matched.iloc[0][ORDINAL_TARGET_DEFAULT_INCLUDE_COLUMN]), 1)

    def test_ordinal_target_builder_rejects_unmatched_review_rows(self) -> None:
        dataset = expand_dataset_for_split_tests(
            build_synthetic_dataset_with_rule_benchmark()["dataset"]
        )
        clinician_reviews = pd.DataFrame(
            [
                {
                    "modeling__row_id": "subject_id=999|encounter_id=hadm:999|medication_standardized=foo|review_timestamp=2125-03-20 00:00:00",
                    "subject_id": 999,
                    "encounter_id": "hadm:999",
                    "medication_standardized": "foo",
                    "review_timestamp": "2125-03-20 00:00:00",
                    "review_submission_id": "review-x",
                    "review_version": 1,
                    "label__clinician_priority_level": "high",
                    "label__clinician_review_status": "reviewed",
                }
            ]
        )

        with self.assertRaisesRegex(Exception, "align to Dataset v1|reconciled to Dataset v1"):
            build_first_scope_ordinal_targets(
                encounter_medication_dataset=dataset,
                latest_clinician_reviews=clinician_reviews,
                splits=None,
            )

    def test_write_artifacts_persists_outputs(self) -> None:
        dataset = expand_dataset_for_split_tests(
            build_synthetic_dataset_with_rule_benchmark()["dataset"]
        )
        clinician_reviews = pd.DataFrame(
            [
                {
                    "modeling__row_id": self._row_id(dataset.iloc[0]),
                    "subject_id": dataset.iloc[0]["subject_id"],
                    "encounter_id": dataset.iloc[0]["encounter_id"],
                    "medication_standardized": dataset.iloc[0]["medication_standardized"],
                    "review_timestamp": dataset.iloc[0]["review_timestamp"],
                    "review_submission_id": "review-1",
                    "review_version": 1,
                    "label__clinician_priority_level": "high",
                    "label__clinician_review_status": "reviewed",
                }
            ]
        )
        result = build_first_scope_ordinal_targets(
            encounter_medication_dataset=dataset,
            latest_clinician_reviews=clinician_reviews,
            splits=None,
        )

        with tempfile.TemporaryDirectory() as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            paths = write_first_scope_ordinal_target_artifacts(
                result=result,
                output_path=temp_dir / "ordinal_targets.parquet",
                summary_path=temp_dir / "ordinal_targets_summary.json",
                report_path=temp_dir / "ordinal_targets_report.md",
            )

            self.assertTrue(paths["targets"].exists())
            self.assertTrue(paths["summary"].exists())
            self.assertTrue(paths["report"].exists())

    @staticmethod
    def _row_id(row: pd.Series) -> str:
        return (
            f"subject_id={row['subject_id']}|"
            f"encounter_id={row['encounter_id']}|"
            f"medication_standardized={row['medication_standardized']}|"
            f"review_timestamp={row['review_timestamp']}"
        )
