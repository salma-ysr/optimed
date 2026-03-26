"""Focused tests for the minimal ordinal supervised experiment."""

from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from opti_med.modeling.first_scope_dataset import FIRST_SCOPE_SPLITS_CONTRACT_VERSION
from opti_med.modeling.first_scope_ordinal_baseline import (
    build_first_scope_ordinal_supervised_data,
    run_first_scope_ordinal_baseline,
)
from opti_med.modeling.first_scope_ordinal_targets import build_first_scope_ordinal_targets
from tests.test_first_scope_dataset_v1 import (
    build_synthetic_dataset_with_rule_benchmark,
    expand_dataset_for_split_tests,
)


class FirstScopeOrdinalBaselineTests(unittest.TestCase):
    def test_supervised_data_uses_only_default_included_rows(self) -> None:
        dataset = expand_dataset_for_split_tests(
            build_synthetic_dataset_with_rule_benchmark()["dataset"]
        )
        dataset["feature__age_context"] = list(range(70, 70 + len(dataset)))
        ordinal_targets = _make_ordinal_targets(dataset)
        result = build_first_scope_ordinal_targets(
            encounter_medication_dataset=dataset,
            latest_clinician_reviews=ordinal_targets,
            splits=_make_manual_splits(dataset),
        )

        supervised = build_first_scope_ordinal_supervised_data(
            encounter_medication_dataset=dataset,
            ordinal_targets=result.targets,
        )

        self.assertEqual(len(supervised.rows), 8)
        self.assertTrue(
            supervised.rows["target__default_training_inclusion_flag"].eq(1).all()
        )
        self.assertTrue(supervised.feature_columns)
        self.assertTrue(
            all(column.startswith("feature__") for column in supervised.feature_columns)
        )

    def test_ordinal_baseline_runs_and_is_deterministic(self) -> None:
        dataset = expand_dataset_for_split_tests(
            build_synthetic_dataset_with_rule_benchmark()["dataset"]
        )
        dataset["feature__age_context"] = list(range(70, 70 + len(dataset)))
        result = build_first_scope_ordinal_targets(
            encounter_medication_dataset=dataset,
            latest_clinician_reviews=_make_ordinal_targets(dataset),
            splits=_make_manual_splits(dataset),
        )
        supervised = build_first_scope_ordinal_supervised_data(
            encounter_medication_dataset=dataset,
            ordinal_targets=result.targets,
        )

        first = run_first_scope_ordinal_baseline(
            supervised_data=supervised,
            dataset_path=Path("dataset.parquet"),
            ordinal_targets_path=Path("ordinal_targets.parquet"),
            random_seed=17,
        )
        second = run_first_scope_ordinal_baseline(
            supervised_data=supervised,
            dataset_path=Path("dataset.parquet"),
            ordinal_targets_path=Path("ordinal_targets.parquet"),
            random_seed=17,
        )

        self.assertEqual(first.run_id, second.run_id)
        self.assertEqual(first.metrics, second.metrics)
        pd.testing.assert_frame_equal(first.predictions, second.predictions)
        self.assertIn("feasibility run", first.report_markdown)
        self.assertEqual(
            first.metrics["estimators"]["logistic_regression_multinomial"]["status"],
            "fit",
        )


def _make_ordinal_targets(dataset: pd.DataFrame) -> pd.DataFrame:
    levels = ["low", "medium", "high", "low", "medium", "high", "low", "medium", "high"]
    statuses = ["reviewed"] * 8 + ["uncertain"]
    scores = [2.0, 5.0, 8.0, 1.0, 6.0, 7.0, 3.0, 4.0, None]
    records = []
    for index, row in dataset.reset_index(drop=True).iterrows():
        records.append(
            {
                "modeling__row_id": (
                    f"subject_id={row['subject_id']}|encounter_id={row['encounter_id']}|"
                    f"medication_standardized={row['medication_standardized']}|"
                    f"review_timestamp={row['review_timestamp']}"
                ),
                "subject_id": row["subject_id"],
                "encounter_id": row["encounter_id"],
                "medication_standardized": row["medication_standardized"],
                "review_timestamp": row["review_timestamp"],
                "review_submission_id": f"review-{index}",
                "review_submission_timestamp": f"2125-03-21T11:{index:02d}:00Z",
                "review_version": 1,
                "reviewer_id": "pharm-a",
                "label__clinician_priority_level": levels[index],
                "label__clinician_priority_score": scores[index],
                "label__clinician_priority_score_level": levels[index] if scores[index] is not None else None,
                "label__clinician_review_status": statuses[index],
                "label__clinician_reason_tags": ["polypharmacy"],
            }
        )
    return pd.DataFrame(records)


def _make_manual_splits(dataset: pd.DataFrame) -> pd.DataFrame:
    partitions = [
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "validation",
        "test",
        "test",
    ]
    split_frame = dataset.loc[:, ["subject_id", "encounter_id", "medication_standardized", "review_timestamp"]].copy()
    split_frame["split_partition"] = partitions
    split_frame["split_random_seed"] = 17
    split_frame["split_stratification_label_name"] = "target__ordinal_priority_level_index"
    split_frame["split_subject_stratum"] = "synthetic"
    split_frame["split_config_json"] = "{}"
    split_frame["split_build_run_id"] = "synthetic-splits"
    split_frame["split_contract_version"] = FIRST_SCOPE_SPLITS_CONTRACT_VERSION
    return split_frame
