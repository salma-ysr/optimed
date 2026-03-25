"""Focused tests for the Phase 3 first-scope baseline scaffold."""

from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from opti_med.evaluation.contracts import TrainValidationTestSplitConfig
from opti_med.modeling.first_scope_baseline import (
    ROW_ID_COLUMN,
    _resolve_level_comparison_availability,
    build_first_scope_baseline_supervised_data,
    run_first_scope_baseline,
)
from opti_med.modeling.first_scope_dataset import build_first_scope_splits
from tests.test_first_scope_dataset_v1 import (
    PRIMARY_NEGATIVE_LABEL,
    build_synthetic_dataset_with_rule_benchmark,
    expand_dataset_for_split_tests,
)


class FirstScopeBaselineModelingTests(unittest.TestCase):
    def test_supervised_matrix_uses_only_feature_columns_and_preserves_benchmarks(self) -> None:
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

        supervised_data = build_first_scope_baseline_supervised_data(
            encounter_medication_dataset=dataset,
            splits=splits,
            feature_list_payload=build_synthetic_dataset_with_rule_benchmark()[
                "feature_list"
            ],
        )

        eligible_count = int(
            pd.to_numeric(
                dataset["meta__dataset_row_eligible_for_training_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .sum()
        )
        self.assertEqual(len(supervised_data.rows), eligible_count)
        self.assertTrue(supervised_data.feature_columns)
        self.assertTrue(supervised_data.benchmark_columns)
        self.assertTrue(
            all(column.startswith("feature__") for column in supervised_data.feature_columns)
        )
        self.assertTrue(
            all(
                column.startswith("benchmark__")
                for column in supervised_data.benchmark_columns
            )
        )
        self.assertFalse(
            set(supervised_data.feature_columns) & set(supervised_data.benchmark_columns)
        )
        self.assertIn(ROW_ID_COLUMN, supervised_data.rows.columns)
        self.assertTrue(supervised_data.rows[ROW_ID_COLUMN].is_unique)

    def test_pipeline_skips_non_viable_estimators_when_train_has_single_class(self) -> None:
        dataset = expand_dataset_for_split_tests(
            build_synthetic_dataset_with_rule_benchmark()["dataset"]
        )
        dataset.loc[
            dataset["label__primary_action_label"] == PRIMARY_NEGATIVE_LABEL,
            "feature__age_context",
        ] = 77
        splits = _manual_subject_safe_splits(
            dataset=dataset,
            train_subjects={1005, 1006, 1007},
            validation_subjects={1001, 1002},
            test_subjects={1003, 1004, 1008},
        )

        supervised_data = build_first_scope_baseline_supervised_data(
            encounter_medication_dataset=dataset,
            splits=splits,
        )
        result = run_first_scope_baseline(
            supervised_data=supervised_data,
            dataset_path=Path("dataset.parquet"),
            splits_path=Path("splits.parquet"),
            feature_list_path=None,
            random_seed=17,
        )

        self.assertEqual(
            result.metrics["estimators"]["dummy_majority"]["status"],
            "fit",
        )
        self.assertEqual(
            result.metrics["estimators"]["dummy_stratified"]["status"],
            "skipped",
        )
        self.assertEqual(
            result.metrics["estimators"]["logistic_regression_l2"]["status"],
            "skipped",
        )
        self.assertIn(
            "requires both classes",
            result.metrics["estimators"]["logistic_regression_l2"]["skip_reason"],
        )

    def test_pipeline_reports_unpopulated_current_rule_benchmark_honestly(self) -> None:
        dataset = expand_dataset_for_split_tests(
            build_synthetic_dataset_with_rule_benchmark()["dataset"]
        )
        for column_name in [
            "benchmark__current_rule_score",
            "benchmark__current_rule_label",
            "benchmark__current_rule_summary_alert",
            "benchmark__current_rule_explanation",
        ]:
            dataset[column_name] = pd.NA
        dataset["benchmark__current_rule_available_flag"] = 0
        dataset.loc[
            dataset["label__primary_action_label"] == PRIMARY_NEGATIVE_LABEL,
            "feature__age_context",
        ] = 77
        splits = _manual_subject_safe_splits(
            dataset=dataset,
            train_subjects={1001, 1002, 1005, 1006},
            validation_subjects={1003, 1007},
            test_subjects={1004, 1008},
        )

        supervised_data = build_first_scope_baseline_supervised_data(
            encounter_medication_dataset=dataset,
            splits=splits,
        )
        result = run_first_scope_baseline(
            supervised_data=supervised_data,
            dataset_path=Path("dataset.parquet"),
            splits_path=Path("splits.parquet"),
            feature_list_path=None,
            random_seed=17,
        )

        self.assertEqual(
            result.metrics["benchmarks"]["benchmark_current_rule_score"]["status"],
            "unpopulated",
        )
        self.assertIn(
            "baseline plumbing milestone",
            result.report_markdown,
        )
        self.assertIn(
            "no valid benchmark comparison is available",
            result.report_markdown,
        )
        self.assertIn(
            "Level agreement is now the primary clinical comparison lens",
            result.report_markdown,
        )

    def test_pipeline_skips_logistic_when_only_two_training_positives_exist(self) -> None:
        dataset = expand_dataset_for_split_tests(
            build_synthetic_dataset_with_rule_benchmark()["dataset"]
        )
        dataset.loc[
            dataset["label__primary_action_label"] == PRIMARY_NEGATIVE_LABEL,
            "feature__age_context",
        ] = 77
        splits = _manual_subject_safe_splits(
            dataset=dataset,
            train_subjects={1001, 1005, 1006, 1007},
            validation_subjects={1002, 1003},
            test_subjects={1004, 1008},
        )

        supervised_data = build_first_scope_baseline_supervised_data(
            encounter_medication_dataset=dataset,
            splits=splits,
        )
        result = run_first_scope_baseline(
            supervised_data=supervised_data,
            dataset_path=Path("dataset.parquet"),
            splits_path=Path("splits.parquet"),
            feature_list_path=None,
            random_seed=17,
        )

        self.assertEqual(
            result.metrics["estimators"]["logistic_regression_l2"]["status"],
            "skipped",
        )
        self.assertIn(
            "at least three training rows in the minority class",
            result.metrics["estimators"]["logistic_regression_l2"]["skip_reason"],
        )

    def test_pipeline_outputs_are_deterministic_for_same_inputs(self) -> None:
        dataset = expand_dataset_for_split_tests(
            build_synthetic_dataset_with_rule_benchmark()["dataset"]
        )
        dataset.loc[
            dataset["label__primary_action_label"] == PRIMARY_NEGATIVE_LABEL,
            "feature__age_context",
        ] = 77
        splits = _manual_subject_safe_splits(
            dataset=dataset,
            train_subjects={1001, 1002, 1005, 1006},
            validation_subjects={1003, 1007},
            test_subjects={1004, 1008},
        )

        supervised_data = build_first_scope_baseline_supervised_data(
            encounter_medication_dataset=dataset,
            splits=splits,
        )
        first_result = run_first_scope_baseline(
            supervised_data=supervised_data,
            dataset_path=Path("dataset.parquet"),
            splits_path=Path("splits.parquet"),
            feature_list_path=None,
            random_seed=17,
        )
        second_result = run_first_scope_baseline(
            supervised_data=supervised_data,
            dataset_path=Path("dataset.parquet"),
            splits_path=Path("splits.parquet"),
            feature_list_path=None,
            random_seed=17,
        )

        self.assertEqual(first_result.run_id, second_result.run_id)
        self.assertEqual(first_result.metrics, second_result.metrics)
        pd.testing.assert_frame_equal(first_result.predictions, second_result.predictions)
        pd.testing.assert_frame_equal(
            first_result.level_comparison_details,
            second_result.level_comparison_details,
        )
        self.assertEqual(first_result.report_markdown, second_result.report_markdown)

    def test_pipeline_coerces_numeric_like_string_features_deterministically(self) -> None:
        dataset = expand_dataset_for_split_tests(
            build_synthetic_dataset_with_rule_benchmark()["dataset"]
        )
        eligible_mask = (
            pd.to_numeric(
                dataset["meta__dataset_row_eligible_for_training_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .eq(1)
        )
        dataset["feature__age_context"] = dataset["feature__age_context"].astype(object)
        dataset.loc[eligible_mask, "feature__age_context"] = (
            dataset.loc[eligible_mask, "feature__age_context"].astype(str)
        )
        splits = _manual_subject_safe_splits(
            dataset=dataset,
            train_subjects={1001, 1002, 1005, 1006, 1007},
            validation_subjects={1003},
            test_subjects={1004, 1008},
        )

        supervised_data = build_first_scope_baseline_supervised_data(
            encounter_medication_dataset=dataset,
            splits=splits,
        )
        result = run_first_scope_baseline(
            supervised_data=supervised_data,
            dataset_path=Path("dataset.parquet"),
            splits_path=Path("splits.parquet"),
            feature_list_path=None,
            random_seed=17,
        )

        self.assertEqual(result.metrics["estimators"]["dummy_majority"]["status"], "fit")
        self.assertIn("modeling__row_id", result.predictions.columns)
        self.assertIn("benchmark__current_rule_score_level", result.predictions.columns)

    def test_level_comparison_rejects_mismatched_semantic_families(self) -> None:
        estimator_rows = pd.DataFrame(
            {
                "prediction__priority_score": [8],
                "prediction__priority_score_level": ["high"],
                "label__clinician_priority_score": [8],
                "label__clinician_priority_score_level": ["high"],
            }
        )
        availability = _resolve_level_comparison_availability(
            estimator_rows=estimator_rows,
            left_source_name="prediction_priority_score",
            right_source_name="label_clinician_priority_score",
            left_spec={
                "score_column": "prediction__priority_score",
                "level_column": "prediction__priority_score_level",
                "semantic_family": "canonical_priority_score_0_to_10",
                "score_column_present": True,
                "level_column_present": True,
                "source_description": "left",
            },
            right_spec={
                "score_column": "label__clinician_priority_score",
                "level_column": "label__clinician_priority_score_level",
                "semantic_family": "different_contract",
                "score_column_present": True,
                "level_column_present": True,
                "source_description": "right",
            },
        )
        self.assertEqual(availability["status"], "unavailable")
        self.assertIn("semantic family", availability["status_reason"])


def _manual_subject_safe_splits(
    *,
    dataset: pd.DataFrame,
    train_subjects: set[int],
    validation_subjects: set[int],
    test_subjects: set[int],
) -> pd.DataFrame:
    """Create a validated split artifact by reassigning persisted subject partitions."""
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
    splits = splits.copy()
    subject_to_partition = {
        **{subject_id: "train" for subject_id in train_subjects},
        **{subject_id: "validation" for subject_id in validation_subjects},
        **{subject_id: "test" for subject_id in test_subjects},
    }
    splits["split_partition"] = splits["subject_id"].map(subject_to_partition)
    return splits


if __name__ == "__main__":
    unittest.main()
