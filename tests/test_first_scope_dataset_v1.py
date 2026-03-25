"""Synthetic tests for Dataset v1 and subject-safe splits."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from opti_med.data_access.artifact_schemas import (
    ENCOUNTER_MEDICATION_BURDEN_COLUMNS,
    ENCOUNTER_MEDICATION_SEMANTICS_COLUMNS,
)
from opti_med.evaluation.contracts import TrainValidationTestSplitConfig
from opti_med.features.first_scope_feature_store import (
    FIRST_SCOPE_FEATURE_STORE_COLUMNS,
    FIRST_SCOPE_FEATURE_STORE_CONTRACT_VERSION,
)
from opti_med.labels.first_scope import build_first_scope_labels
from opti_med.modeling.first_scope_dataset import (
    CANONICAL_TRAINING_FILTER,
    PRIMARY_NEGATIVE_LABEL,
    PRIMARY_POSITIVE_LABEL,
    build_first_scope_dataset,
    build_first_scope_dataset_qc_report,
    build_first_scope_splits,
    select_benchmark_columns_from_dataset,
    select_trainable_feature_columns_from_dataset,
    validate_first_scope_dataset_artifact,
    validate_first_scope_split_artifact,
)
from opti_med.scoring.priority_levels import derive_priority_level_series
from tests.test_first_scope_labels_v1 import (
    make_encounter_index,
    make_first_scope,
    make_mapping,
    make_medication_events,
    make_state,
)


class FirstScopeDatasetV1Tests(unittest.TestCase):
    def test_dataset_builder_separates_trainable_labels_and_benchmarks(self) -> None:
        dataset = build_synthetic_dataset_with_rule_benchmark()

        validate_first_scope_dataset_artifact(dataset["dataset"])
        row = dataset["dataset"].iloc[0]

        self.assertIn("feature__age_context", dataset["dataset"].columns)
        self.assertIn("label__primary_action_label", dataset["dataset"].columns)
        self.assertIn(
            "benchmark__medication_class_only_medication_class_standardized",
            dataset["dataset"].columns,
        )
        self.assertIn(
            "benchmark__polypharmacy_only_exact_current_medication_count",
            dataset["dataset"].columns,
        )
        self.assertIn("benchmark__current_rule_score", dataset["dataset"].columns)
        self.assertEqual(
            row["label__primary_action_label"],
            PRIMARY_POSITIVE_LABEL,
        )
        self.assertEqual(int(row["label__primary_action_binary"]), 1)
        self.assertEqual(int(row["label__trainable_primary_supervision_flag"]), 1)
        self.assertEqual(int(row["meta__dataset_row_eligible_for_training_flag"]), 1)
        self.assertAlmostEqual(float(row["benchmark__current_rule_score"]), 8.0, places=6)
        self.assertEqual(row["benchmark__current_rule_score_level"], "high")
        self.assertEqual(int(row["benchmark__current_rule_available_flag"]), 1)

        feature_list = dataset["feature_list"]
        self.assertIn("feature__age_context", feature_list["feature_columns"])
        self.assertIn("label__primary_action_binary", feature_list["label_columns"])
        self.assertIn(
            "benchmark__current_rule_score",
            feature_list["benchmark_only_columns"],
        )
        self.assertIn(
            "meta__dataset_row_eligible_for_training_flag",
            feature_list["meta_columns"],
        )

    def test_split_builder_is_subject_safe(self) -> None:
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

        validate_first_scope_split_artifact(splits)
        self.assertEqual(
            set(splits["split_partition"].astype(str).tolist()),
            {"train", "validation", "test"},
        )
        self.assertTrue(
            splits.groupby("subject_id")["split_partition"].nunique().eq(1).all()
        )
        self.assertEqual(
            int(splits.loc[splits["subject_id"] == 1001, "split_partition"].nunique()),
            1,
        )
        self.assertEqual(
            int(splits["subject_id"].nunique(dropna=True)),
            8,
        )

    def test_dataset_and_split_build_ids_are_deterministic_for_same_inputs(self) -> None:
        fixture = build_synthetic_dataset_with_rule_benchmark()
        first_result = build_first_scope_dataset(
            encounter_medication_first_scope=fixture["inputs"]["first_scope"],
            encounter_medication_semantics=fixture["inputs"]["semantics"],
            encounter_medication_burden=fixture["inputs"]["burden"],
            first_scope_feature_store=fixture["inputs"]["feature_store"],
            encounter_medication_labels=fixture["inputs"]["labels"],
            polypharmacy_threshold=5,
            benchmark_scores=fixture["inputs"]["benchmark_scores"],
        )
        second_result = build_first_scope_dataset(
            encounter_medication_first_scope=fixture["inputs"]["first_scope"],
            encounter_medication_semantics=fixture["inputs"]["semantics"],
            encounter_medication_burden=fixture["inputs"]["burden"],
            first_scope_feature_store=fixture["inputs"]["feature_store"],
            encounter_medication_labels=fixture["inputs"]["labels"],
            polypharmacy_threshold=5,
            benchmark_scores=fixture["inputs"]["benchmark_scores"],
        )

        self.assertEqual(
            first_result.dataset["meta__dataset_build_run_id"].iloc[0],
            second_result.dataset["meta__dataset_build_run_id"].iloc[0],
        )

        first_splits = build_first_scope_splits(
            encounter_medication_dataset=expand_dataset_for_split_tests(
                first_result.dataset
            ),
            split_config=TrainValidationTestSplitConfig(
                train_fraction=0.50,
                validation_fraction=0.25,
                test_fraction=0.25,
                random_seed=17,
                stratification_label_name="label__primary_action_binary",
            ),
        )
        second_splits = build_first_scope_splits(
            encounter_medication_dataset=expand_dataset_for_split_tests(
                second_result.dataset
            ),
            split_config=TrainValidationTestSplitConfig(
                train_fraction=0.50,
                validation_fraction=0.25,
                test_fraction=0.25,
                random_seed=17,
                stratification_label_name="label__primary_action_binary",
            ),
        )
        self.assertEqual(
            first_splits["split_build_run_id"].iloc[0],
            second_splits["split_build_run_id"].iloc[0],
        )

    def test_dataset_qc_report_mentions_shape_prevalence_and_leakage(self) -> None:
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

        report = build_first_scope_dataset_qc_report(
            encounter_medication_dataset=dataset,
            splits=splits,
        )

        self.assertIn("## Dataset Shape", report)
        self.assertIn("- rows: 9", report)
        self.assertIn("- unique subjects: 8", report)
        self.assertIn("- unique medications: 9", report)
        self.assertIn("## Class Mix", report)
        self.assertIn("## Label Prevalence", report)
        self.assertIn("## Trainable Row Label Prevalence", report)
        self.assertIn(PRIMARY_POSITIVE_LABEL, report)
        self.assertIn(PRIMARY_NEGATIVE_LABEL, report)
        self.assertIn("canonical training row filter", report)
        self.assertIn("`benchmark__*` columns are preserved", report)
        self.assertIn("leakage_check_passed: True", report)

    def test_dataset_selectors_keep_feature_and_benchmark_columns_separate(self) -> None:
        dataset = build_synthetic_dataset_with_rule_benchmark()["dataset"]

        feature_columns = select_trainable_feature_columns_from_dataset(dataset)
        benchmark_columns = select_benchmark_columns_from_dataset(dataset)

        self.assertTrue(feature_columns)
        self.assertTrue(benchmark_columns)
        self.assertTrue(all(column.startswith("feature__") for column in feature_columns))
        self.assertTrue(
            all(column.startswith("benchmark__") for column in benchmark_columns)
        )
        self.assertEqual(
            CANONICAL_TRAINING_FILTER["meta__dataset_row_eligible_for_training_flag"],
            1,
        )


def build_synthetic_dataset_with_rule_benchmark() -> dict[str, object]:
    """Build a one-row Dataset v1 artifact from synthetic upstream inputs."""
    first_scope = make_first_scope([{}])
    semantics = first_scope.loc[:, ENCOUNTER_MEDICATION_SEMANTICS_COLUMNS].copy()
    burden = first_scope.loc[:, ENCOUNTER_MEDICATION_BURDEN_COLUMNS].copy()
    labels = build_first_scope_labels(
        encounter_medication_first_scope=first_scope,
        encounter_medication_state=make_state(first_scope),
        medication_events=make_medication_events(
            [
                {
                    "medication_event_id": "order-review",
                    "medication_event_type": "hospital_order",
                    "raw_medication_name": "Oxycodone 10 mg",
                    "medication_name": "Oxycodone 10 mg",
                    "medication_normalized": "oxycodone",
                    "event_time": "2125-03-20 08:00:00",
                    "starttime": "2125-03-20 08:00:00",
                    "stoptime": "2125-03-20 14:00:00",
                    "route": "PO",
                    "frequency": "BID",
                    "status": "active",
                    "dose_value": "10",
                    "dose_unit": "mg",
                }
            ]
        ),
        medication_rxnorm_mapping=make_mapping(first_scope),
        encounter_index=make_encounter_index(),
    )
    feature_store = make_feature_store(first_scope)
    benchmark_scores = pd.DataFrame(
        [
            {
                "subject_id": 101,
                "encounter_id": "hadm:1",
                "medication_standardized": "oxycodone",
                "review_timestamp": "2125-03-20 10:00:00",
                "deprescribing_priority_score": 8,
                "deprescribing_priority_label": "high",
                "deprescribing_priority_summary_alert": "legacy rule benchmark",
                "deprescribing_priority_explanation": "synthetic baseline only",
            }
        ]
    )
    result = build_first_scope_dataset(
        encounter_medication_first_scope=first_scope,
        encounter_medication_semantics=semantics,
        encounter_medication_burden=burden,
        first_scope_feature_store=feature_store,
        encounter_medication_labels=labels,
        polypharmacy_threshold=5,
        benchmark_scores=benchmark_scores,
    )
    return {
        "dataset": result.dataset,
        "feature_list": result.feature_list,
        "inputs": {
            "first_scope": first_scope,
            "semantics": semantics,
            "burden": burden,
            "labels": labels,
            "feature_store": feature_store,
            "benchmark_scores": benchmark_scores,
        },
    }


def make_feature_store(first_scope: pd.DataFrame) -> pd.DataFrame:
    """Build a minimally valid Feature Store v1 artifact for dataset tests."""
    records: list[dict[str, object]] = []
    for row in first_scope.to_dict(orient="records"):
        record = {column_name: None for column_name in FIRST_SCOPE_FEATURE_STORE_COLUMNS}
        record.update(
            {
                "subject_id": row["subject_id"],
                "encounter_id": row["encounter_id"],
                "medication_standardized": row["medication_standardized"],
                "review_timestamp": row["review_timestamp"],
                "hadm_id": row["hadm_id"],
                "stay_id": row["stay_id"],
                "review_timestamp_source": row["review_timestamp_source"],
                "medication_class_standardized": row["medication_class_standardized"],
                "medication_standardized_source": row["medication_standardized_source"],
                "ingredient_standardized": row["ingredient_standardized"],
                "ingredient_resolution_status": row["ingredient_resolution_status"],
                "mapping_confidence": row["mapping_confidence"],
                "ambiguous_mapping_flag": row["ambiguous_mapping_flag"],
                "mapping_candidate_count": row["mapping_candidate_count"],
                "active_at_review_flag": row["active_at_review_flag"],
                "medication_status_at_review": row["medication_status_at_review"],
                "continued_from_home_inferred": row["continued_from_home_inferred"],
                "newly_started_during_encounter_inferred": row[
                    "newly_started_during_encounter_inferred"
                ],
                "medication_start_context": row["medication_start_context"],
                "duration_before_review_hours": row["duration_before_review_hours"],
                "duration_before_review_inferable_flag": row[
                    "duration_before_review_inferable_flag"
                ],
                "duration_before_review_lower_bound_flag": row[
                    "duration_before_review_lower_bound_flag"
                ],
                "scheduled_vs_prn": row["scheduled_vs_prn"],
                "scheduled_vs_prn_inference_status": row[
                    "scheduled_vs_prn_inference_status"
                ],
                "dose_value": row["dose_value"],
                "dose_unit": row["dose_unit"],
                "route": row["route"],
                "frequency": row["frequency"],
                "exact_current_medication_count": row["exact_current_medication_count"],
                "current_benzodiazepine_count": row["current_benzodiazepine_count"],
                "current_opioid_count": row["current_opioid_count"],
                "current_anticholinergic_count": row[
                    "current_anticholinergic_count"
                ],
                "current_ppi_count": row["current_ppi_count"],
                "current_antipsychotic_count": row["current_antipsychotic_count"],
                "current_supported_class_count": row["current_supported_class_count"],
                "row_same_class_current_count": row["row_same_class_current_count"],
                "same_class_duplicate_therapy_flag": row[
                    "same_class_duplicate_therapy_flag"
                ],
                "same_class_duplicate_therapy_signal_count": row[
                    "same_class_duplicate_therapy_signal_count"
                ],
                "review_time_validated_flag": 1,
                "review_time_policy_name": "discharge_capped_latest_available",
                "review_time_capped_to_discharge_flag": 0,
                "source_home_medrecon_flag": 0,
                "source_ed_pyxis_flag": 0,
                "source_hospital_order_flag": 1,
                "source_hospital_admin_flag": 0,
                "age_context": row["age_proxy"],
                "age_group": row["age_group"],
                "sex_context": "F",
                "sex_provenance": "observed_patients_gender",
                "weight_kg": 70.0,
                "weight_kg_provenance": "observed_omr",
                "bmi": 25.0,
                "bmi_provenance": "observed_omr",
                "egfr_ml_min_1_73m2": 64.0,
                "egfr_provenance": "derived_egfr_ckd_epi_2021",
                "cockcroft_gault_ml_min": 58.0,
                "cockcroft_gault_provenance": "derived_cockcroft_gault",
                "prior_hospital_admission_count_all_time": 1,
                "prior_hospital_admission_count_365d": 0,
                "prior_ed_stay_count_all_time": 0,
                "prior_ed_stay_count_365d": 0,
                "prior_utilization_provenance": "synthetic",
                "prior_utilization_window_definition": "all_time_and_365d_pre_review",
                "prior_ckd_flag": 0,
                "prior_dementia_flag": 0,
                "prior_delirium_flag": 0,
                "prior_heart_failure_flag": 0,
                "prior_diabetes_flag": 0,
                "prior_diagnosis_risk_renal_flag": 0,
                "prior_diagnosis_risk_cognitive_flag": 0,
                "prior_diagnosis_risk_cardiac_flag": 0,
                "prior_diagnosis_risk_metabolic_flag": 0,
                "prior_diagnosis_context_provenance": "synthetic",
                "ed_ckd_flag": 0,
                "ed_dementia_flag": 0,
                "ed_delirium_flag": 0,
                "ed_heart_failure_flag": 0,
                "ed_diabetes_flag": 0,
                "ed_diagnosis_risk_renal_flag": 0,
                "ed_diagnosis_risk_cognitive_flag": 0,
                "ed_diagnosis_risk_cardiac_flag": 0,
                "ed_diagnosis_risk_metabolic_flag": 0,
                "ed_diagnosis_context_provenance": "synthetic",
                "ed_diagnosis_context_missingness": "observed",
                "patient_context_window_end": row["review_timestamp"],
                "patient_context_feature_window_end_at_or_before_review_time_flag": 1,
                "patient_context_available_as_of_review_time_flag": 1,
                "patient_context_source_tables_json": json.dumps(["synthetic_context"]),
                "creatinine_observation_count_pre_review": 1,
                "creatinine_first": 1.2,
                "creatinine_last": 1.2,
                "creatinine_min": 1.2,
                "creatinine_max": 1.2,
                "creatinine_mean": 1.2,
                "creatinine_delta": 0.0,
                "creatinine_trend_direction": "stable",
                "creatinine_provenance": "observed_labevents_pre_review",
                "creatinine_missingness": "observed",
                "potassium_observation_count_pre_review": 1,
                "potassium_first": 4.0,
                "potassium_last": 4.0,
                "potassium_min": 4.0,
                "potassium_max": 4.0,
                "potassium_mean": 4.0,
                "potassium_delta": 0.0,
                "potassium_trend_direction": "stable",
                "potassium_provenance": "observed_labevents_pre_review",
                "potassium_missingness": "observed",
                "sodium_observation_count_pre_review": 1,
                "sodium_first": 138.0,
                "sodium_last": 138.0,
                "sodium_min": 138.0,
                "sodium_max": 138.0,
                "sodium_mean": 138.0,
                "sodium_delta": 0.0,
                "sodium_trend_direction": "stable",
                "sodium_provenance": "observed_labevents_pre_review",
                "sodium_missingness": "observed",
                "sbp_observation_count_pre_review": 1,
                "sbp_min": 110.0,
                "sbp_max": 110.0,
                "sbp_mean": 110.0,
                "sbp_last": 110.0,
                "dbp_observation_count_pre_review": 1,
                "dbp_min": 70.0,
                "dbp_max": 70.0,
                "dbp_mean": 70.0,
                "dbp_last": 70.0,
                "heart_rate_observation_count_pre_review": 1,
                "heart_rate_min": 85.0,
                "heart_rate_max": 85.0,
                "heart_rate_mean": 85.0,
                "heart_rate_last": 85.0,
                "pain_observation_count_pre_review": 1,
                "pain_min": 2.0,
                "pain_max": 2.0,
                "pain_mean": 2.0,
                "pain_last": 2.0,
                "lab_window_start": "2125-03-20 08:00:00",
                "lab_window_end": "2125-03-20 09:00:00",
                "lab_window_end_at_or_before_review_time_flag": 1,
                "vital_window_start": "2125-03-20 07:30:00",
                "vital_window_end": "2125-03-20 09:00:00",
                "vital_window_end_at_or_before_review_time_flag": 1,
                "temporal_window_end": "2125-03-20 09:00:00",
                "temporal_feature_window_end_at_or_before_review_time_flag": 1,
                "blood_pressure_provenance": "observed_ed_vitals_pre_review",
                "heart_rate_provenance": "observed_ed_vitals_pre_review",
                "pain_provenance": "observed_ed_vitals_pre_review",
                "temporal_available_as_of_review_time_flag": 1,
                "temporal_source_tables_json": json.dumps(["synthetic_temporal"]),
                "state_source_tables_json": json.dumps(["synthetic_state"]),
                "feature_available_as_of_review_time_flag": 1,
                "feature_window_end_at_or_before_review_time_flag": 1,
                "feature_source_tables_json": json.dumps(
                    ["synthetic_context", "synthetic_temporal", "synthetic_state"]
                ),
                "missingness_profile_json": json.dumps({}),
                "first_scope_feature_store_build_run_id": "feature-store-run",
                "first_scope_feature_store_contract_version": (
                    FIRST_SCOPE_FEATURE_STORE_CONTRACT_VERSION
                ),
            }
        )
        records.append(record)
    return pd.DataFrame(records, columns=FIRST_SCOPE_FEATURE_STORE_COLUMNS)


def expand_dataset_for_split_tests(base_dataset: pd.DataFrame) -> pd.DataFrame:
    """Clone a validated one-row dataset into a multi-subject split fixture."""
    subject_blueprint = [
        (1001, "hadm:1001", "oxycodone-a", PRIMARY_POSITIVE_LABEL),
        (1001, "hadm:1001", "oxycodone-b", PRIMARY_POSITIVE_LABEL),
        (1002, "hadm:1002", "oxycodone-c", PRIMARY_POSITIVE_LABEL),
        (1003, "hadm:1003", "oxycodone-d", PRIMARY_POSITIVE_LABEL),
        (1004, "hadm:1004", "oxycodone-e", PRIMARY_POSITIVE_LABEL),
        (1005, "hadm:1005", "oxycodone-f", PRIMARY_NEGATIVE_LABEL),
        (1006, "hadm:1006", "oxycodone-g", PRIMARY_NEGATIVE_LABEL),
        (1007, "hadm:1007", "oxycodone-h", PRIMARY_NEGATIVE_LABEL),
        (1008, "hadm:1008", "oxycodone-i", "action_undetermined"),
    ]

    cloned_rows: list[pd.Series] = []
    template = base_dataset.iloc[0].copy()
    for offset, (subject_id, encounter_id, medication_name, label_name) in enumerate(
        subject_blueprint,
        start=1,
    ):
        row = template.copy()
        row["subject_id"] = subject_id
        row["encounter_id"] = encounter_id
        row["medication_standardized"] = medication_name
        row["review_timestamp"] = f"2125-03-{20 + offset:02d} 10:00:00"
        row["meta__hadm_id"] = subject_id
        row["meta__stay_id"] = subject_id + 10_000
        row["benchmark__current_rule_score"] = (
            8 if label_name == PRIMARY_POSITIVE_LABEL else 5
        )
        row["benchmark__current_rule_score_level"] = (
            "high" if label_name == PRIMARY_POSITIVE_LABEL else "medium"
        )
        row["benchmark__current_rule_label"] = (
            "high" if label_name == PRIMARY_POSITIVE_LABEL else "medium"
        )
        row["benchmark__current_rule_summary_alert"] = "synthetic benchmark"
        row["benchmark__current_rule_explanation"] = "synthetic benchmark only"
        row["benchmark__current_rule_available_flag"] = 1
        row["label__primary_action_label_reason"] = f"synthetic_{label_name}"
        row["label__primary_action_evidence_json"] = json.dumps({"synthetic": label_name})
        if label_name == PRIMARY_POSITIVE_LABEL:
            row["label__primary_action_binary"] = 1
            row["label__unknown_or_insufficient_evidence_flag"] = 0
            row["label__unknown_or_insufficient_evidence_reason"] = None
            row["label__window_censored_flag"] = 0
            row["label__excluded_from_primary_training"] = 0
            row["label__post_review_stop_evidence_flag"] = 1
            row["label__post_review_continuation_evidence_flag"] = 0
            row["label__trainable_primary_supervision_flag"] = 1
            row["meta__dataset_row_eligible_for_training_flag"] = 1
        elif label_name == PRIMARY_NEGATIVE_LABEL:
            row["label__primary_action_binary"] = 0
            row["label__unknown_or_insufficient_evidence_flag"] = 0
            row["label__unknown_or_insufficient_evidence_reason"] = None
            row["label__window_censored_flag"] = 0
            row["label__excluded_from_primary_training"] = 0
            row["label__post_review_stop_evidence_flag"] = 0
            row["label__post_review_continuation_evidence_flag"] = 1
            row["label__trainable_primary_supervision_flag"] = 1
            row["meta__dataset_row_eligible_for_training_flag"] = 1
        else:
            row["label__primary_action_binary"] = pd.NA
            row["label__unknown_or_insufficient_evidence_flag"] = 1
            row["label__unknown_or_insufficient_evidence_reason"] = (
                "synthetic_unknown_label"
            )
            row["label__window_censored_flag"] = 0
            row["label__excluded_from_primary_training"] = 0
            row["label__post_review_stop_evidence_flag"] = 0
            row["label__post_review_continuation_evidence_flag"] = 0
            row["label__trainable_primary_supervision_flag"] = 0
            row["meta__dataset_row_eligible_for_training_flag"] = 0
        row["label__primary_action_label"] = label_name
        cloned_rows.append(row)

    dataset = pd.DataFrame(cloned_rows, columns=base_dataset.columns)
    if "benchmark__current_rule_score" in dataset.columns:
        dataset["benchmark__current_rule_score_level"] = derive_priority_level_series(
            dataset["benchmark__current_rule_score"]
        )
    dataset = dataset.sort_values(
        ["subject_id", "encounter_id", "medication_standardized", "review_timestamp"]
    ).reset_index(drop=True)
    validate_first_scope_dataset_artifact(dataset)
    return dataset


if __name__ == "__main__":
    unittest.main()
