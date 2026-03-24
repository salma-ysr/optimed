"""Synthetic evaluation contract tests with subject-leakage guardrails only."""

from __future__ import annotations

import unittest

from opti_med.evaluation.contracts import (
    BASELINE_NAMES,
    DEFAULT_BASELINE_NAMES,
    ANALYTICAL_EVALUATION_GRAIN_DESCRIPTION,
    ANALYTICAL_EVALUATION_PRIMARY_KEY,
    BaselineName,
    CalibrationMetric,
    RankingMetric,
    SplitPartition,
    SubjectLeakagePreventionConfig,
    TrainValidationTestSplitConfig,
    assert_no_subject_leakage,
    build_partition_subject_index,
    validate_split_config,
)


class EvaluationContractConstantsTests(unittest.TestCase):
    def test_evaluation_grain_matches_review_time_medication_shape(self) -> None:
        self.assertEqual(
            ANALYTICAL_EVALUATION_GRAIN_DESCRIPTION,
            "one row per subject_id + encounter_id + medication_standardized + review_timestamp",
        )
        self.assertEqual(
            ANALYTICAL_EVALUATION_PRIMARY_KEY,
            ("subject_id", "encounter_id", "medication_standardized", "review_timestamp"),
        )

    def test_baseline_names_are_explicit_and_stable(self) -> None:
        self.assertEqual(
            BASELINE_NAMES,
            (
                "current_rule_score",
                "medication_class_only_baseline",
                "polypharmacy_only_baseline",
            ),
        )
        self.assertEqual(
            DEFAULT_BASELINE_NAMES,
            (
                BaselineName.CURRENT_RULE_SCORE,
                BaselineName.MEDICATION_CLASS_ONLY_BASELINE,
                BaselineName.POLYPHARMACY_ONLY_BASELINE,
            ),
        )

    def test_metric_enums_include_ranking_and_calibration_contracts(self) -> None:
        self.assertIn(RankingMetric.NDCG_AT_K, tuple(RankingMetric))
        self.assertIn(CalibrationMetric.BRIER_SCORE, tuple(CalibrationMetric))


class EvaluationSplitLeakageTests(unittest.TestCase):
    def test_valid_split_config_passes(self) -> None:
        config = TrainValidationTestSplitConfig(
            train_fraction=0.7,
            validation_fraction=0.15,
            test_fraction=0.15,
        )
        validate_split_config(config)

    def test_split_config_rejects_non_subject_grouping(self) -> None:
        config = TrainValidationTestSplitConfig(
            train_fraction=0.7,
            validation_fraction=0.15,
            test_fraction=0.15,
            subject_leakage_prevention=SubjectLeakagePreventionConfig(
                require_subject_level_grouping=False,
            ),
        )
        with self.assertRaises(ValueError):
            validate_split_config(config)

    def test_subject_leakage_assertion_passes_for_disjoint_subject_sets(self) -> None:
        partition_index = build_partition_subject_index(
            train_subject_ids=[1, 2, 3],
            validation_subject_ids=[4, 5],
            test_subject_ids=[6, 7],
        )
        assert_no_subject_leakage(partition_index)

    def test_subject_leakage_assertion_raises_when_subject_overlaps(self) -> None:
        partition_index = build_partition_subject_index(
            train_subject_ids=[1, 2, 3],
            validation_subject_ids=[3, 4],
            test_subject_ids=[5, 6],
        )
        with self.assertRaises(ValueError):
            assert_no_subject_leakage(partition_index)

    def test_partition_index_uses_stable_partition_keys(self) -> None:
        partition_index = build_partition_subject_index(
            train_subject_ids=[11],
            validation_subject_ids=[22],
            test_subject_ids=[33],
        )
        self.assertEqual(
            set(partition_index.keys()),
            {SplitPartition.TRAIN, SplitPartition.VALIDATION, SplitPartition.TEST},
        )


if __name__ == "__main__":
    unittest.main()
