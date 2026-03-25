"""Dataset v1 assembly and subject-safe splits for the 65+ first-scope branch."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from random import Random

import pandas as pd

from opti_med.contracts.pipeline import (
    ANALYTICAL_GRAIN_DESCRIPTION,
    ANALYTICAL_GRAIN_PRIMARY_KEY,
)
from opti_med.data_access.artifact_schemas import (
    validate_encounter_medication_burden_artifact,
    validate_encounter_medication_first_scope_artifact,
    validate_encounter_medication_labels_v1_first_scope_artifact,
    validate_encounter_medication_semantics_artifact,
)
from opti_med.data_access.exceptions import DataLoadError
from opti_med.data_access.provenance import dumps_json
from opti_med.evaluation.contracts import (
    SplitGenerator,
    SplitPartition,
    TrainValidationTestSplitConfig,
    assert_no_subject_leakage,
    build_partition_subject_index,
    validate_split_config,
)
from opti_med.features.first_scope_feature_store import (
    validate_first_scope_feature_store_artifact,
)
from opti_med.scoring.priority_levels import (
    derive_priority_level_series,
    validate_score_level_alignment,
)


FIRST_SCOPE_DATASET_CONTRACT_VERSION = (
    "encounter_medication_dataset_v1_65plus_first_scope.v1"
)
FIRST_SCOPE_SPLITS_CONTRACT_VERSION = "splits_v1_65plus_first_scope.v1"
DATASET_KEY_COLUMNS = list(ANALYTICAL_GRAIN_PRIMARY_KEY)
PRIMARY_POSITIVE_LABEL = "stopped_or_deintensified_before_discharge"
PRIMARY_NEGATIVE_LABEL = "no_clear_stop_or_deintensification_before_discharge"
BENCHMARK_SCORE_RENAME_CANDIDATES: tuple[dict[str, str], ...] = (
    {
        "current_rule_score": "current_rule_score",
        "current_rule_label": "current_rule_label",
        "current_rule_summary_alert": "current_rule_summary_alert",
        "current_rule_explanation": "current_rule_explanation",
    },
    {
        "deprescribing_priority_score": "current_rule_score",
        "deprescribing_priority_label": "current_rule_label",
        "deprescribing_priority_summary_alert": "current_rule_summary_alert",
        "deprescribing_priority_explanation": "current_rule_explanation",
    },
)
FEATURE_EXACT_EXCLUSIONS = {
    "hadm_id",
    "stay_id",
    "review_timestamp_source",
    "review_time_validated_flag",
    "review_time_policy_name",
    "review_time_capped_to_discharge_flag",
    "source_home_medrecon_flag",
    "source_ed_pyxis_flag",
    "source_hospital_order_flag",
    "source_hospital_admin_flag",
    "feature_available_as_of_review_time_flag",
    "feature_window_end_at_or_before_review_time_flag",
    "state_source_tables_json",
    "feature_source_tables_json",
    "missingness_profile_json",
    "first_scope_feature_store_build_run_id",
    "first_scope_feature_store_contract_version",
}
FEATURE_SUFFIX_EXCLUSIONS = (
    "_provenance",
    "_missingness",
    "_unavailable_reason",
    "_source_tables_json",
    "_build_run_id",
    "_contract_version",
    "_window_start",
    "_window_end",
    "_latest_time",
    "_time",
)
FEATURE_CONTAINS_EXCLUSIONS = (
    "_at_or_before_review_time_flag",
    "_available_as_of_review_time_flag",
)
SEMANTICS_BENCHMARK_COLUMNS = [
    "medication_class_standardized",
    "benzodiazepine_heuristic_flag",
    "opioid_heuristic_flag",
    "anticholinergic_heuristic_flag",
    "ppi_heuristic_flag",
    "antipsychotic_heuristic_flag",
    "heuristic_any_supported_class_flag",
    "standardized_vs_heuristic_class_agreement_flag",
]
BURDEN_BENCHMARK_COLUMNS = ["exact_current_medication_count"]
LABEL_DATASET_COLUMNS = [
    "primary_action_label",
    "primary_action_label_reason",
    "unknown_or_insufficient_evidence_flag",
    "unknown_or_insufficient_evidence_reason",
    "window_censored_flag",
    "renal_deterioration",
    "hemodynamic_instability",
    "electrolyte_instability",
    "oversedation_respiratory_risk",
    "acute_life_sustaining_medication",
    "fundamentally_different_deprescribing_logic",
    "excluded_from_primary_training",
    "downweight_recommended",
    "post_review_same_medication_event_count",
    "post_review_same_medication_order_count",
    "post_review_same_medication_admin_count",
    "post_review_stop_evidence_flag",
    "post_review_deintensification_evidence_flag",
    "post_review_continuation_evidence_flag",
    "primary_action_evidence_json",
    "auxiliary_harm_evidence_json",
    "exclusion_evidence_json",
    "label_window_json",
    "label_provenance_json",
    "label_source_tables_json",
]
SPLIT_COLUMNS = [
    *DATASET_KEY_COLUMNS,
    "split_partition",
    "split_random_seed",
    "split_stratification_label_name",
    "split_subject_stratum",
    "split_config_json",
    "split_build_run_id",
    "split_contract_version",
]
CANONICAL_TRAINING_FILTER = {
    "label__trainable_primary_supervision_flag": 1,
    "meta__feature_available_as_of_review_time_flag": 1,
    "meta__dataset_row_eligible_for_training_flag": 1,
}


@dataclass(frozen=True, slots=True)
class FirstScopeDatasetBuildResult:
    """Built Dataset v1 and its feature-list payload."""

    dataset: pd.DataFrame
    feature_list: dict[str, object]


class StableSubjectSplitGenerator(SplitGenerator):
    """Deterministic subject-level splitter with optional subject strata."""

    def __init__(self, *, subject_strata: dict[int, str] | None = None) -> None:
        self.subject_strata = subject_strata or {}

    def generate_subject_partitions(
        self,
        subject_ids,
        config: TrainValidationTestSplitConfig,
    ) -> dict[SplitPartition, set[int]]:
        validate_split_config(config)
        unique_subject_ids = sorted({int(subject_id) for subject_id in subject_ids})
        rng = Random(config.random_seed)
        partitions = {
            SplitPartition.TRAIN: set(),
            SplitPartition.VALIDATION: set(),
            SplitPartition.TEST: set(),
        }
        if not unique_subject_ids:
            return partitions

        if self.subject_strata:
            strata_to_subjects: dict[str, list[int]] = {}
            for subject_id in unique_subject_ids:
                stratum = self.subject_strata.get(subject_id, "unlabeled")
                strata_to_subjects.setdefault(stratum, []).append(subject_id)
            for subject_list in strata_to_subjects.values():
                rng.shuffle(subject_list)
                counts = _partition_counts(len(subject_list), config=config)
                assignments = _partition_subject_sequence(subject_list, counts)
                for partition, partition_subjects in assignments.items():
                    partitions[partition].update(partition_subjects)
        else:
            shuffled = unique_subject_ids[:]
            rng.shuffle(shuffled)
            counts = _partition_counts(len(shuffled), config=config)
            assignments = _partition_subject_sequence(shuffled, counts)
            for partition, partition_subjects in assignments.items():
                partitions[partition].update(partition_subjects)

        assert_no_subject_leakage(partitions)
        return partitions


def select_trainable_feature_columns_from_dataset(
    encounter_medication_dataset: pd.DataFrame,
) -> list[str]:
    """Return the stable trainable feature columns from a persisted Dataset v1 artifact."""
    validate_first_scope_dataset_artifact(encounter_medication_dataset)
    return [
        column_name
        for column_name in encounter_medication_dataset.columns
        if column_name.startswith("feature__")
    ]


def select_benchmark_columns_from_dataset(
    encounter_medication_dataset: pd.DataFrame,
) -> list[str]:
    """Return benchmark-only columns from a persisted Dataset v1 artifact."""
    validate_first_scope_dataset_artifact(encounter_medication_dataset)
    return [
        column_name
        for column_name in encounter_medication_dataset.columns
        if column_name.startswith("benchmark__")
    ]


def select_label_columns_from_dataset(
    encounter_medication_dataset: pd.DataFrame,
) -> list[str]:
    """Return label columns from a persisted Dataset v1 artifact."""
    validate_first_scope_dataset_artifact(encounter_medication_dataset)
    return [
        column_name
        for column_name in encounter_medication_dataset.columns
        if column_name.startswith("label__")
    ]


def select_meta_columns_from_dataset(
    encounter_medication_dataset: pd.DataFrame,
) -> list[str]:
    """Return meta columns from a persisted Dataset v1 artifact."""
    validate_first_scope_dataset_artifact(encounter_medication_dataset)
    return [
        column_name
        for column_name in encounter_medication_dataset.columns
        if column_name.startswith("meta__")
    ]


def build_first_scope_dataset(
    *,
    encounter_medication_first_scope: pd.DataFrame,
    encounter_medication_semantics: pd.DataFrame,
    encounter_medication_burden: pd.DataFrame,
    first_scope_feature_store: pd.DataFrame,
    encounter_medication_labels: pd.DataFrame,
    polypharmacy_threshold: int,
    benchmark_scores: pd.DataFrame | None = None,
) -> FirstScopeDatasetBuildResult:
    """Assemble Dataset v1 for the 65+ first-scope modeling branch."""
    validate_encounter_medication_first_scope_artifact(encounter_medication_first_scope)
    validate_encounter_medication_semantics_artifact(encounter_medication_semantics)
    validate_encounter_medication_burden_artifact(encounter_medication_burden)
    validate_first_scope_feature_store_artifact(first_scope_feature_store)
    validate_encounter_medication_labels_v1_first_scope_artifact(encounter_medication_labels)

    base = _prepare_first_scope_base_frame(encounter_medication_first_scope)
    semantics = _prepare_semantics_benchmark_frame(encounter_medication_semantics)
    burden = _prepare_burden_benchmark_frame(
        encounter_medication_burden,
        polypharmacy_threshold=polypharmacy_threshold,
    )
    feature_columns = derive_trainable_feature_columns(first_scope_feature_store)
    feature_store = _prepare_feature_store_frame(
        first_scope_feature_store,
        trainable_feature_columns=feature_columns,
    )
    labels = _prepare_label_frame(encounter_medication_labels)
    benchmark_scores_frame = _prepare_benchmark_scores_frame(
        benchmark_scores=benchmark_scores,
        base=base,
    )

    dataset = base.copy()
    for artifact_name, frame in [
        ("encounter_medication_semantics_65plus", semantics),
        ("encounter_medication_burden_65plus", burden),
        ("encounter_medication_features_v1_65plus_first_scope", feature_store),
        ("encounter_medication_labels_v1_65plus_first_scope", labels),
        ("benchmark_scores", benchmark_scores_frame),
    ]:
        dataset = _merge_required_frame(
            base=dataset,
            frame=frame,
            artifact_name=artifact_name,
            require_all_matches=artifact_name != "benchmark_scores",
        )

    primary_label = dataset["label__primary_action_label"].astype(str)
    dataset["label__primary_action_binary"] = primary_label.map(
        {
            PRIMARY_POSITIVE_LABEL: 1,
            PRIMARY_NEGATIVE_LABEL: 0,
        }
    )
    label_trainable = (
        dataset["label__primary_action_binary"].notna()
        & (
            pd.to_numeric(
                dataset["label__excluded_from_primary_training"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            == 0
        )
        & (
            pd.to_numeric(
                dataset["label__unknown_or_insufficient_evidence_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            == 0
        )
        & (
            pd.to_numeric(
                dataset["label__window_censored_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            == 0
        )
    )
    dataset["label__trainable_primary_supervision_flag"] = label_trainable.astype(int)
    dataset["meta__dataset_row_eligible_for_training_flag"] = (
        label_trainable
        & (
            pd.to_numeric(
                dataset["meta__feature_available_as_of_review_time_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            == 1
        )
    ).astype(int)
    build_run_id = _stable_build_run_id(
        prefix="first-scope-dataset",
        payload={
            "dataset_contract_version": FIRST_SCOPE_DATASET_CONTRACT_VERSION,
            "polypharmacy_threshold": int(polypharmacy_threshold),
            "row_count": int(len(dataset)),
            "subject_count": int(dataset["subject_id"].nunique(dropna=True)),
            "encounter_count": int(dataset["encounter_id"].nunique(dropna=True)),
            "feature_columns": [
                column_name.removeprefix("feature__")
                for column_name in sorted(_select_prefixed_columns(dataset, "feature__"))
            ],
            "base_build_ids": _distinct_text_values(
                dataset["meta__encounter_medication_first_scope_build_run_id"]
            ),
            "semantics_build_ids": _distinct_text_values(
                dataset["meta__encounter_medication_semantics_build_run_id"]
            ),
            "burden_build_ids": _distinct_text_values(
                dataset["meta__encounter_medication_burden_build_run_id"]
            ),
            "feature_store_build_ids": _distinct_text_values(
                dataset["meta__first_scope_feature_store_build_run_id"]
            ),
            "label_build_ids": _distinct_text_values(
                dataset["meta__encounter_medication_labels_v1_build_run_id"]
            ),
            "benchmark_current_rule_available_row_count": int(
                pd.to_numeric(
                    dataset["benchmark__current_rule_available_flag"],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
                .sum()
            ),
        },
    )
    dataset["meta__dataset_build_run_id"] = build_run_id
    dataset["meta__dataset_contract_version"] = FIRST_SCOPE_DATASET_CONTRACT_VERSION

    dataset = dataset.sort_values(DATASET_KEY_COLUMNS, na_position="last").reset_index(drop=True)
    dataset = _order_first_scope_dataset_columns(dataset)
    validate_first_scope_dataset_artifact(dataset)
    feature_list = build_first_scope_feature_list(dataset)
    return FirstScopeDatasetBuildResult(dataset=dataset, feature_list=feature_list)


def derive_trainable_feature_columns(first_scope_feature_store: pd.DataFrame) -> list[str]:
    """Derive the stable trainable feature column list from Feature Store v1."""
    validate_first_scope_feature_store_artifact(first_scope_feature_store)
    feature_columns: list[str] = []
    for column_name in first_scope_feature_store.columns:
        if column_name in DATASET_KEY_COLUMNS:
            continue
        if column_name in FEATURE_EXACT_EXCLUSIONS:
            continue
        if any(column_name.endswith(suffix) for suffix in FEATURE_SUFFIX_EXCLUSIONS):
            continue
        if any(fragment in column_name for fragment in FEATURE_CONTAINS_EXCLUSIONS):
            continue
        feature_columns.append(column_name)
    return feature_columns


def build_first_scope_feature_list(dataset: pd.DataFrame) -> dict[str, object]:
    """Build the persisted feature-list JSON payload for Dataset v1."""
    validate_first_scope_dataset_artifact(dataset)
    feature_columns = select_trainable_feature_columns_from_dataset(dataset)
    label_columns = select_label_columns_from_dataset(dataset)
    benchmark_columns = select_benchmark_columns_from_dataset(dataset)
    meta_columns = select_meta_columns_from_dataset(dataset)
    return {
        "artifact_name": "encounter_medication_dataset_v1_65plus_first_scope",
        "dataset_contract_version": FIRST_SCOPE_DATASET_CONTRACT_VERSION,
        "grain_description": ANALYTICAL_GRAIN_DESCRIPTION,
        "primary_key": DATASET_KEY_COLUMNS,
        "feature_columns": feature_columns,
        "label_columns": label_columns,
        "benchmark_only_columns": benchmark_columns,
        "meta_columns": meta_columns,
        "feature_column_count": len(feature_columns),
        "label_column_count": len(label_columns),
        "benchmark_only_column_count": len(benchmark_columns),
        "row_count": int(len(dataset)),
        "unique_subject_count": int(dataset["subject_id"].nunique(dropna=True)),
        "unique_encounter_count": int(dataset["encounter_id"].nunique(dropna=True)),
        "unique_medication_count": int(
            dataset["medication_standardized"].nunique(dropna=True)
        ),
        "eligible_training_row_count": int(
            pd.to_numeric(
                dataset["meta__dataset_row_eligible_for_training_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .sum()
        ),
        "recommended_training_filter": dict(CANONICAL_TRAINING_FILTER),
        "recommended_training_filter_expression": (
            "meta__dataset_row_eligible_for_training_flag == 1"
        ),
        "column_family_separation": {
            "trainable_feature_prefix": "feature__",
            "benchmark_only_prefix": "benchmark__",
            "label_prefix": "label__",
            "meta_prefix": "meta__",
        },
        "notes": [
            "feature__ columns are the trainable feature set for Dataset v1.",
            "label__ columns carry primary and auxiliary supervision plus label evidence/provenance.",
            "benchmark__ columns are preserved for baseline comparison only and should not be mixed into learned-feature selection by default.",
            "meta__ columns carry identifiers, artifact lineage, and point-in-time availability auditing.",
            "The canonical row-eligibility filter for future training is meta__dataset_row_eligible_for_training_flag == 1.",
        ],
    }


def validate_first_scope_dataset_artifact(dataframe: pd.DataFrame) -> None:
    """Validate Dataset v1 for the 65+ first-scope modeling branch."""
    required_columns = [
        *DATASET_KEY_COLUMNS,
        "meta__hadm_id",
        "meta__stay_id",
        "meta__review_timestamp_source",
        "meta__feature_available_as_of_review_time_flag",
        "meta__dataset_row_eligible_for_training_flag",
        "meta__dataset_build_run_id",
        "meta__dataset_contract_version",
        "label__primary_action_label",
        "label__primary_action_binary",
        "label__excluded_from_primary_training",
        "label__unknown_or_insufficient_evidence_flag",
        "label__window_censored_flag",
        "label__trainable_primary_supervision_flag",
        "benchmark__medication_class_only_medication_class_standardized",
        "benchmark__polypharmacy_only_exact_current_medication_count",
        "benchmark__polypharmacy_only_threshold",
        "benchmark__polypharmacy_only_flag",
        "benchmark__current_rule_available_flag",
    ]
    missing_columns = sorted(set(required_columns) - set(dataframe.columns))
    if missing_columns:
        raise DataLoadError(
            "Dataset v1 artifact is missing required columns: "
            + ", ".join(missing_columns)
        )
    duplicate_mask = dataframe.duplicated(subset=DATASET_KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        raise DataLoadError(
            "Dataset v1 artifact has duplicate analytical-grain rows."
        )
    if not any(column.startswith("feature__") for column in dataframe.columns):
        raise DataLoadError("Dataset v1 artifact must include at least one feature__ column.")
    for column_name in [
        "meta__feature_available_as_of_review_time_flag",
        "meta__dataset_row_eligible_for_training_flag",
        "label__excluded_from_primary_training",
        "label__unknown_or_insufficient_evidence_flag",
        "label__window_censored_flag",
        "label__trainable_primary_supervision_flag",
        "benchmark__polypharmacy_only_flag",
        "benchmark__current_rule_available_flag",
    ]:
        _assert_binary_column(dataframe, column_name, artifact_name="dataset_v1")
    allowed_labels = {
        PRIMARY_POSITIVE_LABEL,
        PRIMARY_NEGATIVE_LABEL,
        "action_undetermined",
        "window_censored",
    }
    unexpected_labels = sorted(
        {
            str(value)
            for value in dataframe["label__primary_action_label"].dropna().astype(str).tolist()
            if str(value) not in allowed_labels
        }
    )
    if unexpected_labels:
        raise DataLoadError(
            "Dataset v1 artifact has unsupported primary labels: "
            + ", ".join(unexpected_labels)
        )
    binary_values = pd.to_numeric(
        dataframe["label__primary_action_binary"],
        errors="coerce",
    )
    non_null_binary = binary_values[dataframe["label__primary_action_binary"].notna()]
    if non_null_binary.isna().any() or not set(non_null_binary.astype(int).tolist()).issubset({0, 1}):
        raise DataLoadError(
            "Dataset v1 artifact has non-binary values in label__primary_action_binary."
        )
    expected_binary = dataframe["label__primary_action_label"].map(
        {
            PRIMARY_POSITIVE_LABEL: 1,
            PRIMARY_NEGATIVE_LABEL: 0,
        }
    )
    binary_match = expected_binary.fillna(-1).eq(binary_values.fillna(-1))
    if not binary_match.all():
        raise DataLoadError(
            "Dataset v1 artifact must keep label__primary_action_binary aligned with label__primary_action_label."
        )
    expected_trainable = (
        expected_binary.notna()
        & (
            pd.to_numeric(
                dataframe["label__excluded_from_primary_training"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            == 0
        )
        & (
            pd.to_numeric(
                dataframe["label__unknown_or_insufficient_evidence_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            == 0
        )
        & (
            pd.to_numeric(
                dataframe["label__window_censored_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            == 0
        )
    ).astype(int)
    actual_trainable = pd.to_numeric(
        dataframe["label__trainable_primary_supervision_flag"],
        errors="coerce",
    ).fillna(0).astype(int)
    if not actual_trainable.eq(expected_trainable).all():
        raise DataLoadError(
            "Dataset v1 artifact must keep label__trainable_primary_supervision_flag aligned with primary-label eligibility."
        )
    expected_row_eligibility = (
        expected_trainable.eq(1)
        & (
            pd.to_numeric(
                dataframe["meta__feature_available_as_of_review_time_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            == 1
        )
    ).astype(int)
    actual_row_eligibility = pd.to_numeric(
        dataframe["meta__dataset_row_eligible_for_training_flag"],
        errors="coerce",
    ).fillna(0).astype(int)
    if not actual_row_eligibility.eq(expected_row_eligibility).all():
        raise DataLoadError(
            "Dataset v1 artifact must keep meta__dataset_row_eligible_for_training_flag aligned with label eligibility and feature availability."
        )
    poly_count = pd.to_numeric(
        dataframe["benchmark__polypharmacy_only_exact_current_medication_count"],
        errors="coerce",
    )
    poly_threshold = pd.to_numeric(
        dataframe["benchmark__polypharmacy_only_threshold"],
        errors="coerce",
    )
    poly_flag = pd.to_numeric(
        dataframe["benchmark__polypharmacy_only_flag"],
        errors="coerce",
    ).fillna(0).astype(int)
    if poly_count.isna().any() or poly_threshold.isna().any():
        raise DataLoadError(
            "Dataset v1 artifact must keep polypharmacy-only benchmark inputs non-null."
        )
    if not poly_flag.eq((poly_count >= poly_threshold).astype(int)).all():
        raise DataLoadError(
            "Dataset v1 artifact must keep benchmark__polypharmacy_only_flag aligned with the count and threshold."
        )
    if (
        dataframe["meta__dataset_contract_version"].dropna().astype(str)
        != FIRST_SCOPE_DATASET_CONTRACT_VERSION
    ).any():
        raise DataLoadError(
            "Dataset v1 artifact must use contract version "
            f"'{FIRST_SCOPE_DATASET_CONTRACT_VERSION}'."
        )
    if "benchmark__current_rule_score_level" in dataframe.columns:
        try:
            validate_score_level_alignment(
                score_series=dataframe["benchmark__current_rule_score"],
                level_series=dataframe["benchmark__current_rule_score_level"],
                score_name="benchmark__current_rule_score",
                level_name="benchmark__current_rule_score_level",
            )
        except ValueError as exc:
            raise DataLoadError(str(exc)) from exc


def build_first_scope_splits(
    *,
    encounter_medication_dataset: pd.DataFrame,
    split_config: TrainValidationTestSplitConfig,
) -> pd.DataFrame:
    """Generate subject-safe row-level split assignments for Dataset v1."""
    validate_first_scope_dataset_artifact(encounter_medication_dataset)
    validate_split_config(split_config)
    subject_strata = _subject_strata_from_dataset(
        encounter_medication_dataset,
        label_column=split_config.stratification_label_name,
    )
    generator = StableSubjectSplitGenerator(subject_strata=subject_strata)
    partitions = generator.generate_subject_partitions(
        subject_ids=encounter_medication_dataset["subject_id"].tolist(),
        config=split_config,
    )
    partition_lookup = {
        subject_id: partition.value
        for partition, subject_ids in partitions.items()
        for subject_id in subject_ids
    }
    stratum_lookup = {
        subject_id: subject_strata.get(subject_id, "unlabeled")
        for subject_id in partition_lookup
    }
    config_json = dumps_json(
        {
            "train_fraction": split_config.train_fraction,
            "validation_fraction": split_config.validation_fraction,
            "test_fraction": split_config.test_fraction,
            "random_seed": split_config.random_seed,
            "stratification_label_name": split_config.stratification_label_name,
        }
    )
    build_run_id = _stable_build_run_id(
        prefix="first-scope-splits",
        payload={
            "splits_contract_version": FIRST_SCOPE_SPLITS_CONTRACT_VERSION,
            "dataset_build_run_id": _single_stable_text_value(
                encounter_medication_dataset["meta__dataset_build_run_id"]
            ),
            "dataset_row_count": int(len(encounter_medication_dataset)),
            "subject_count": int(
                encounter_medication_dataset["subject_id"].nunique(dropna=True)
            ),
            "config_json": config_json,
        },
    )
    splits = encounter_medication_dataset.loc[:, DATASET_KEY_COLUMNS].copy()
    splits["split_partition"] = splits["subject_id"].map(partition_lookup)
    splits["split_random_seed"] = (
        split_config.random_seed if split_config.random_seed is not None else -1
    )
    splits["split_stratification_label_name"] = (
        split_config.stratification_label_name or "unstratified"
    )
    splits["split_subject_stratum"] = splits["subject_id"].map(stratum_lookup).fillna("unlabeled")
    splits["split_config_json"] = config_json
    splits["split_build_run_id"] = build_run_id
    splits["split_contract_version"] = FIRST_SCOPE_SPLITS_CONTRACT_VERSION
    splits = splits.loc[:, SPLIT_COLUMNS].copy()
    splits = splits.sort_values(DATASET_KEY_COLUMNS, na_position="last").reset_index(drop=True)
    validate_first_scope_split_artifact(splits)
    return splits


def validate_first_scope_split_artifact(dataframe: pd.DataFrame) -> None:
    """Validate row-level subject-safe split assignments for Dataset v1."""
    missing_columns = sorted(set(SPLIT_COLUMNS) - set(dataframe.columns))
    if missing_columns:
        raise DataLoadError(
            "Split artifact is missing required columns: " + ", ".join(missing_columns)
        )
    duplicate_mask = dataframe.duplicated(subset=DATASET_KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        raise DataLoadError("Split artifact has duplicate analytical-grain assignments.")
    allowed_partitions = {partition.value for partition in SplitPartition}
    unexpected_values = sorted(
        {
            str(value)
            for value in dataframe["split_partition"].dropna().astype(str).tolist()
            if str(value) not in allowed_partitions
        }
    )
    if unexpected_values:
        raise DataLoadError(
            "Split artifact has unsupported split_partition values: "
            + ", ".join(unexpected_values)
        )
    _assert_non_null_column(dataframe, "split_build_run_id", artifact_name="splits_v1")
    _assert_non_null_column(dataframe, "split_contract_version", artifact_name="splits_v1")
    partition_index = build_partition_subject_index(
        train_subject_ids=dataframe.loc[dataframe["split_partition"] == SplitPartition.TRAIN.value, "subject_id"].astype(int).tolist(),
        validation_subject_ids=dataframe.loc[dataframe["split_partition"] == SplitPartition.VALIDATION.value, "subject_id"].astype(int).tolist(),
        test_subject_ids=dataframe.loc[dataframe["split_partition"] == SplitPartition.TEST.value, "subject_id"].astype(int).tolist(),
    )
    assert_no_subject_leakage(partition_index)
    if (
        dataframe["split_contract_version"].dropna().astype(str)
        != FIRST_SCOPE_SPLITS_CONTRACT_VERSION
    ).any():
        raise DataLoadError(
            "Split artifact must use contract version "
            f"'{FIRST_SCOPE_SPLITS_CONTRACT_VERSION}'."
        )


def write_first_scope_dataset_qc_report(
    *,
    encounter_medication_dataset: pd.DataFrame,
    splits: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write the Dataset v1 QC report to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_first_scope_dataset_qc_report(
            encounter_medication_dataset=encounter_medication_dataset,
            splits=splits,
        ),
        encoding="utf-8",
    )
    return output_path


def build_first_scope_dataset_qc_report(
    *,
    encounter_medication_dataset: pd.DataFrame,
    splits: pd.DataFrame,
) -> str:
    """Render a Markdown QC report for Dataset v1 and its splits."""
    metrics = calculate_first_scope_dataset_qc_metrics(
        encounter_medication_dataset=encounter_medication_dataset,
        splits=splits,
    )
    lines: list[str] = [
        "# Dataset V1 65plus First-Scope QC",
        "",
        "## Artifact Lineage",
        "",
        f"- dataset_build_run_id: {metrics['dataset_build_run_id']}",
        f"- split_build_run_id: {metrics['split_build_run_id']}",
        "",
        "## Dataset Shape",
        "",
        f"- rows: {metrics['row_count']:,}",
        f"- unique subjects: {metrics['unique_subject_count']:,}",
        f"- unique encounters: {metrics['unique_encounter_count']:,}",
        f"- unique medications: {metrics['unique_medication_count']:,}",
        f"- eligible training rows: {metrics['eligible_training_row_count']:,}",
        f"- trainable feature columns: {metrics['feature_column_count']:,}",
        f"- benchmark-only columns: {metrics['benchmark_column_count']:,}",
        "",
        "## Class Mix",
        "",
    ]
    if metrics["class_mix"]:
        for class_name, count in metrics["class_mix"].items():
            lines.append(f"- {class_name}: {count:,}")
    else:
        lines.append("- no rows")
    lines.extend(
        [
            "",
            "## Label Prevalence",
            "",
        ]
    )
    for label_name, count in metrics["label_prevalence"].items():
        rate = metrics["label_prevalence_rate"][label_name]
        lines.append(f"- {label_name}: {count:,} ({rate:.2%})")
    lines.extend(
        [
            "",
            "## Trainable Row Label Prevalence",
            "",
        ]
    )
    for label_name, count in metrics["trainable_label_prevalence"].items():
        rate = metrics["trainable_label_prevalence_rate"][label_name]
        lines.append(f"- {label_name}: {count:,} ({rate:.2%})")
    lines.extend(
        [
            "",
            "## Split Summary",
            "",
        ]
    )
    for partition_name, payload in metrics["partition_summary"].items():
        lines.append(
            f"- {partition_name}: rows={payload['row_count']:,}, "
            f"subjects={payload['subject_count']:,}, "
            f"eligible_training_rows={payload['eligible_training_row_count']:,}, "
            f"trainable_labels={payload['trainable_label_count']:,}"
        )
    lines.extend(
        [
            "",
            "## Training Interface",
            "",
            "- canonical training row filter: `meta__dataset_row_eligible_for_training_flag == 1`",
            "- trainable inputs are only the `feature__*` columns.",
            "- `benchmark__*` columns are preserved for comparison and fallback only; they are not training inputs.",
            "",
            "## Subject Leakage Checks",
            "",
            f"- train/validation overlap subjects: {metrics['subject_leakage_overlaps']['train_validation']:,}",
            f"- train/test overlap subjects: {metrics['subject_leakage_overlaps']['train_test']:,}",
            f"- validation/test overlap subjects: {metrics['subject_leakage_overlaps']['validation_test']:,}",
            f"- leakage_check_passed: {metrics['subject_leakage_check_passed']}",
        ]
    )
    return "\n".join(lines) + "\n"


def calculate_first_scope_dataset_qc_metrics(
    *,
    encounter_medication_dataset: pd.DataFrame,
    splits: pd.DataFrame,
) -> dict[str, object]:
    """Calculate compact QC metrics for Dataset v1 and its split assignments."""
    validate_first_scope_dataset_artifact(encounter_medication_dataset)
    validate_first_scope_split_artifact(splits)
    _assert_split_alignment(
        encounter_medication_dataset=encounter_medication_dataset,
        splits=splits,
    )
    class_column = (
        "feature__medication_class_standardized"
        if "feature__medication_class_standardized" in encounter_medication_dataset
        else "benchmark__medication_class_only_medication_class_standardized"
    )
    class_mix = {
        str(key): int(value)
        for key, value in encounter_medication_dataset[class_column]
        .fillna("null")
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
        .items()
    }
    label_prevalence = {
        str(key): int(value)
        for key, value in encounter_medication_dataset["label__primary_action_label"]
        .fillna("null")
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
        .items()
    }
    eligible_training_mask = (
        pd.to_numeric(
            encounter_medication_dataset["meta__dataset_row_eligible_for_training_flag"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        == 1
    )
    trainable_subset = encounter_medication_dataset.loc[eligible_training_mask].copy()
    trainable_label_prevalence = {
        str(key): int(value)
        for key, value in trainable_subset["label__primary_action_label"]
        .fillna("null")
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
        .items()
    }
    row_count = int(len(encounter_medication_dataset))
    partition_summary: dict[str, dict[str, int]] = {}
    merged = encounter_medication_dataset.merge(
        splits.loc[:, DATASET_KEY_COLUMNS + ["split_partition"]],
        how="left",
        on=DATASET_KEY_COLUMNS,
        validate="one_to_one",
    )
    for partition in SplitPartition:
        subset = merged.loc[merged["split_partition"] == partition.value].copy()
        partition_summary[partition.value] = {
            "row_count": int(len(subset)),
            "subject_count": int(subset["subject_id"].nunique(dropna=True)),
            "eligible_training_row_count": int(
                pd.to_numeric(
                    subset["meta__dataset_row_eligible_for_training_flag"],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
                .sum()
            ),
            "trainable_label_count": int(
                pd.to_numeric(
                    subset["label__trainable_primary_supervision_flag"],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
                .sum()
            ),
        }
    train_subjects = set(
        merged.loc[merged["split_partition"] == SplitPartition.TRAIN.value, "subject_id"]
        .dropna()
        .astype(int)
        .tolist()
    )
    validation_subjects = set(
        merged.loc[
            merged["split_partition"] == SplitPartition.VALIDATION.value,
            "subject_id",
        ]
        .dropna()
        .astype(int)
        .tolist()
    )
    test_subjects = set(
        merged.loc[merged["split_partition"] == SplitPartition.TEST.value, "subject_id"]
        .dropna()
        .astype(int)
        .tolist()
    )
    overlaps = {
        "train_validation": int(len(train_subjects & validation_subjects)),
        "train_test": int(len(train_subjects & test_subjects)),
        "validation_test": int(len(validation_subjects & test_subjects)),
    }
    return {
        "row_count": row_count,
        "unique_subject_count": int(encounter_medication_dataset["subject_id"].nunique(dropna=True)),
        "unique_encounter_count": int(encounter_medication_dataset["encounter_id"].nunique(dropna=True)),
        "unique_medication_count": int(
            encounter_medication_dataset["medication_standardized"].nunique(dropna=True)
        ),
        "eligible_training_row_count": int(
            pd.to_numeric(
                encounter_medication_dataset["meta__dataset_row_eligible_for_training_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .sum()
        ),
        "class_mix": class_mix,
        "label_prevalence": label_prevalence,
        "label_prevalence_rate": {
            label_name: (count / row_count) if row_count else 0.0
            for label_name, count in label_prevalence.items()
        },
        "trainable_label_prevalence": trainable_label_prevalence,
        "trainable_label_prevalence_rate": {
            label_name: (
                count / int(len(trainable_subset)) if len(trainable_subset) else 0.0
            )
            for label_name, count in trainable_label_prevalence.items()
        },
        "partition_summary": partition_summary,
        "feature_column_count": len(
            select_trainable_feature_columns_from_dataset(encounter_medication_dataset)
        ),
        "benchmark_column_count": len(
            select_benchmark_columns_from_dataset(encounter_medication_dataset)
        ),
        "dataset_build_run_id": _single_stable_text_value(
            encounter_medication_dataset["meta__dataset_build_run_id"]
        ),
        "split_build_run_id": _single_stable_text_value(splits["split_build_run_id"]),
        "subject_leakage_overlaps": overlaps,
        "subject_leakage_check_passed": all(count == 0 for count in overlaps.values()),
    }


def summarize_first_scope_dataset(
    *,
    encounter_medication_dataset: pd.DataFrame,
) -> list[str]:
    """Return compact summary lines for Dataset v1."""
    validate_first_scope_dataset_artifact(encounter_medication_dataset)
    label_counts = ", ".join(
        f"{label_name}={count:,}"
        for label_name, count in encounter_medication_dataset["label__primary_action_label"]
        .fillna("null")
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
        .items()
    ) or "none"
    feature_count = len(
        select_trainable_feature_columns_from_dataset(encounter_medication_dataset)
    )
    return [
        (
            f"rows={len(encounter_medication_dataset):,}, "
            f"subjects={encounter_medication_dataset['subject_id'].nunique(dropna=True):,}, "
            f"encounters={encounter_medication_dataset['encounter_id'].nunique(dropna=True):,}"
        ),
        f"trainable_feature_columns={feature_count:,}",
        f"primary_label_counts={label_counts}",
    ]


def _prepare_first_scope_base_frame(
    encounter_medication_first_scope: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        *DATASET_KEY_COLUMNS,
        "hadm_id",
        "stay_id",
        "review_timestamp_source",
        "encounter_medication_first_scope_build_run_id",
        "encounter_medication_first_scope_contract_version",
    ]
    return encounter_medication_first_scope.loc[:, columns].rename(
        columns={
            "hadm_id": "meta__hadm_id",
            "stay_id": "meta__stay_id",
            "review_timestamp_source": "meta__review_timestamp_source",
            "encounter_medication_first_scope_build_run_id": (
                "meta__encounter_medication_first_scope_build_run_id"
            ),
            "encounter_medication_first_scope_contract_version": (
                "meta__encounter_medication_first_scope_contract_version"
            ),
        }
    )


def _order_first_scope_dataset_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    """Apply the canonical column-family ordering for Dataset v1."""
    ordered_columns = [
        *DATASET_KEY_COLUMNS,
        *select_meta_columns_from_dataset(dataset),
        *select_label_columns_from_dataset(dataset),
        *select_benchmark_columns_from_dataset(dataset),
        *select_trainable_feature_columns_from_dataset(dataset),
    ]
    remaining_columns = [
        column_name
        for column_name in dataset.columns
        if column_name not in ordered_columns
    ]
    return dataset.loc[:, [*ordered_columns, *remaining_columns]].copy()


def _prepare_semantics_benchmark_frame(
    encounter_medication_semantics: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        *DATASET_KEY_COLUMNS,
        *SEMANTICS_BENCHMARK_COLUMNS,
        "encounter_medication_semantics_build_run_id",
        "encounter_medication_semantics_contract_version",
    ]
    frame = encounter_medication_semantics.loc[:, columns].copy()
    rename_map = {
        "medication_class_standardized": (
            "benchmark__medication_class_only_medication_class_standardized"
        ),
        "benzodiazepine_heuristic_flag": "benchmark__benzodiazepine_heuristic_flag",
        "opioid_heuristic_flag": "benchmark__opioid_heuristic_flag",
        "anticholinergic_heuristic_flag": "benchmark__anticholinergic_heuristic_flag",
        "ppi_heuristic_flag": "benchmark__ppi_heuristic_flag",
        "antipsychotic_heuristic_flag": "benchmark__antipsychotic_heuristic_flag",
        "heuristic_any_supported_class_flag": "benchmark__heuristic_any_supported_class_flag",
        "standardized_vs_heuristic_class_agreement_flag": (
            "benchmark__standardized_vs_heuristic_class_agreement_flag"
        ),
        "encounter_medication_semantics_build_run_id": (
            "meta__encounter_medication_semantics_build_run_id"
        ),
        "encounter_medication_semantics_contract_version": (
            "meta__encounter_medication_semantics_contract_version"
        ),
    }
    return frame.rename(columns=rename_map)


def _prepare_burden_benchmark_frame(
    encounter_medication_burden: pd.DataFrame,
    *,
    polypharmacy_threshold: int,
) -> pd.DataFrame:
    columns = [
        *DATASET_KEY_COLUMNS,
        *BURDEN_BENCHMARK_COLUMNS,
        "encounter_medication_burden_build_run_id",
        "encounter_medication_burden_contract_version",
    ]
    frame = encounter_medication_burden.loc[:, columns].copy()
    frame["benchmark__polypharmacy_only_exact_current_medication_count"] = pd.to_numeric(
        frame["exact_current_medication_count"],
        errors="coerce",
    )
    frame["benchmark__polypharmacy_only_threshold"] = int(polypharmacy_threshold)
    frame["benchmark__polypharmacy_only_flag"] = (
        frame["benchmark__polypharmacy_only_exact_current_medication_count"] >= polypharmacy_threshold
    ).astype(int)
    frame["meta__encounter_medication_burden_build_run_id"] = frame[
        "encounter_medication_burden_build_run_id"
    ]
    frame["meta__encounter_medication_burden_contract_version"] = frame[
        "encounter_medication_burden_contract_version"
    ]
    return frame.loc[
        :,
        [
            *DATASET_KEY_COLUMNS,
            "benchmark__polypharmacy_only_exact_current_medication_count",
            "benchmark__polypharmacy_only_threshold",
            "benchmark__polypharmacy_only_flag",
            "meta__encounter_medication_burden_build_run_id",
            "meta__encounter_medication_burden_contract_version",
        ],
    ].copy()


def _prepare_feature_store_frame(
    first_scope_feature_store: pd.DataFrame,
    *,
    trainable_feature_columns: list[str],
) -> pd.DataFrame:
    excluded_base_meta = {"hadm_id", "stay_id", "review_timestamp_source"}
    meta_columns = [
        column_name
        for column_name in first_scope_feature_store.columns
        if column_name not in DATASET_KEY_COLUMNS
        and column_name not in trainable_feature_columns
        and column_name not in excluded_base_meta
    ]
    selected = first_scope_feature_store.loc[
        :,
        [*DATASET_KEY_COLUMNS, *trainable_feature_columns, *meta_columns],
    ].copy()
    rename_map = {
        column_name: f"feature__{column_name}" for column_name in trainable_feature_columns
    }
    rename_map.update({column_name: f"meta__{column_name}" for column_name in meta_columns})
    return selected.rename(columns=rename_map)


def _prepare_label_frame(
    encounter_medication_labels: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        *DATASET_KEY_COLUMNS,
        *LABEL_DATASET_COLUMNS,
        "encounter_medication_labels_v1_build_run_id",
        "encounter_medication_labels_v1_contract_version",
    ]
    frame = encounter_medication_labels.loc[:, columns].copy()
    rename_map = {
        column_name: f"label__{column_name}" for column_name in LABEL_DATASET_COLUMNS
    }
    rename_map.update(
        {
            "encounter_medication_labels_v1_build_run_id": (
                "meta__encounter_medication_labels_v1_build_run_id"
            ),
            "encounter_medication_labels_v1_contract_version": (
                "meta__encounter_medication_labels_v1_contract_version"
            ),
        }
    )
    return frame.rename(columns=rename_map)


def _prepare_benchmark_scores_frame(
    *,
    benchmark_scores: pd.DataFrame | None,
    base: pd.DataFrame,
) -> pd.DataFrame:
    output = _normalize_join_keys(base.loc[:, DATASET_KEY_COLUMNS].copy())
    output["benchmark__current_rule_score"] = pd.NA
    output["benchmark__current_rule_score_level"] = pd.NA
    output["benchmark__current_rule_label"] = pd.NA
    output["benchmark__current_rule_summary_alert"] = pd.NA
    output["benchmark__current_rule_explanation"] = pd.NA
    output["benchmark__current_rule_available_flag"] = 0
    if benchmark_scores is None:
        return output
    score_frame = _normalize_join_keys(benchmark_scores.copy())
    if not set(DATASET_KEY_COLUMNS).issubset(score_frame.columns):
        raise DataLoadError(
            "Optional benchmark score artifact must expose the analytical key columns "
            "subject_id, encounter_id, medication_standardized, and review_timestamp."
        )
    rename_candidate = _benchmark_score_rename_map(score_frame)
    if rename_candidate is None:
        raise DataLoadError(
            "Optional benchmark score artifact must expose either current_rule_* columns or "
            "legacy deprescribing_priority_* columns at the analytical grain."
        )
    selected_columns = [*DATASET_KEY_COLUMNS, *rename_candidate.keys()]
    score_frame = score_frame.loc[:, selected_columns].rename(columns=rename_candidate)
    duplicate_mask = score_frame.duplicated(subset=DATASET_KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        raise DataLoadError(
            "Optional benchmark score artifact has duplicate analytical-grain keys."
        )
    output = output.merge(
        score_frame,
        how="left",
        on=DATASET_KEY_COLUMNS,
        validate="one_to_one",
        suffixes=("", "_joined"),
    )
    for column_name in [
        "benchmark__current_rule_score",
        "benchmark__current_rule_label",
        "benchmark__current_rule_summary_alert",
        "benchmark__current_rule_explanation",
    ]:
        joined_name = f"{column_name}_joined"
        if joined_name in output.columns:
            output[column_name] = output[joined_name]
            output = output.drop(columns=joined_name)
    output["benchmark__current_rule_available_flag"] = output[
        "benchmark__current_rule_score"
    ].notna().astype(int)
    output["benchmark__current_rule_score_level"] = derive_priority_level_series(
        output["benchmark__current_rule_score"]
    )
    validate_score_level_alignment(
        score_series=output["benchmark__current_rule_score"],
        level_series=output["benchmark__current_rule_score_level"],
        score_name="benchmark__current_rule_score",
        level_name="benchmark__current_rule_score_level",
    )
    return output


def _benchmark_score_rename_map(dataframe: pd.DataFrame) -> dict[str, str] | None:
    for candidate in BENCHMARK_SCORE_RENAME_CANDIDATES:
        present = {source: target for source, target in candidate.items() if source in dataframe.columns}
        if present:
            return {source: f"benchmark__{target}" for source, target in present.items()}
    return None


def _merge_required_frame(
    *,
    base: pd.DataFrame,
    frame: pd.DataFrame,
    artifact_name: str,
    require_all_matches: bool,
) -> pd.DataFrame:
    base_normalized = _normalize_join_keys(base)
    frame_normalized = _normalize_join_keys(frame)
    merged = base_normalized.merge(
        frame_normalized,
        how="left",
        on=DATASET_KEY_COLUMNS,
        validate="one_to_one",
        indicator=True,
    )
    if require_all_matches and not merged["_merge"].eq("both").all():
        missing_count = int((merged["_merge"] != "both").sum())
        raise DataLoadError(
            f"Dataset v1 build could not align {artifact_name} for {missing_count:,} analytical-grain rows."
        )
    return merged.drop(columns="_merge")


def _subject_strata_from_dataset(
    encounter_medication_dataset: pd.DataFrame,
    *,
    label_column: str | None,
) -> dict[int, str]:
    if label_column is None or label_column not in encounter_medication_dataset.columns:
        return {}
    strata: dict[int, str] = {}
    grouped = encounter_medication_dataset.groupby("subject_id", sort=False)
    for subject_id, group in grouped:
        if label_column == "label__primary_action_binary":
            eligible = group.loc[
                pd.to_numeric(
                    group["label__trainable_primary_supervision_flag"],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
                == 1
            ].copy()
            if eligible.empty:
                strata[int(subject_id)] = "unlabeled"
                continue
            values = pd.to_numeric(eligible[label_column], errors="coerce")
            if values.eq(1).any():
                strata[int(subject_id)] = "positive"
            elif values.eq(0).any():
                strata[int(subject_id)] = "negative"
            else:
                strata[int(subject_id)] = "unlabeled"
            continue
        non_null_values = [
            str(value)
            for value in group[label_column].dropna().astype(str).tolist()
            if str(value)
        ]
        strata[int(subject_id)] = non_null_values[0] if non_null_values else "unlabeled"
    return strata


def _select_prefixed_columns(dataframe: pd.DataFrame, prefix: str) -> list[str]:
    """Return stable columns sharing one prefix without validating the full artifact."""
    return [
        column_name
        for column_name in dataframe.columns
        if column_name.startswith(prefix)
    ]


def _stable_build_run_id(*, prefix: str, payload: dict[str, object]) -> str:
    """Build a deterministic artifact run id from stable payload content."""
    digest = hashlib.sha1(dumps_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _distinct_text_values(values: pd.Series) -> list[str]:
    """Return distinct non-empty text values in stable sorted order."""
    series = values.dropna().astype(str)
    return sorted({value for value in series.tolist() if value})


def _single_stable_text_value(values: pd.Series) -> str:
    """Return one stable text value for an artifact-wide metadata column."""
    distinct_values = _distinct_text_values(values)
    if not distinct_values:
        return "missing"
    if len(distinct_values) != 1:
        raise DataLoadError(
            "Expected exactly one stable artifact-level value but found multiple: "
            + ", ".join(distinct_values)
        )
    return distinct_values[0]


def _partition_counts(
    total_subjects: int,
    *,
    config: TrainValidationTestSplitConfig,
) -> dict[SplitPartition, int]:
    raw_counts = {
        SplitPartition.TRAIN: total_subjects * config.train_fraction,
        SplitPartition.VALIDATION: total_subjects * config.validation_fraction,
        SplitPartition.TEST: total_subjects * config.test_fraction,
    }
    base_counts = {
        partition: int(value)
        for partition, value in raw_counts.items()
    }
    remainder = total_subjects - sum(base_counts.values())
    if remainder > 0:
        ranked = sorted(
            raw_counts.items(),
            key=lambda item: (item[1] - int(item[1]), item[1]),
            reverse=True,
        )
        for partition, _ in ranked[:remainder]:
            base_counts[partition] += 1
    return base_counts


def _partition_subject_sequence(
    subject_ids: list[int],
    counts: dict[SplitPartition, int],
) -> dict[SplitPartition, set[int]]:
    train_end = counts[SplitPartition.TRAIN]
    validation_end = train_end + counts[SplitPartition.VALIDATION]
    return {
        SplitPartition.TRAIN: set(subject_ids[:train_end]),
        SplitPartition.VALIDATION: set(subject_ids[train_end:validation_end]),
        SplitPartition.TEST: set(subject_ids[validation_end:]),
    }


def _assert_split_alignment(
    *,
    encounter_medication_dataset: pd.DataFrame,
    splits: pd.DataFrame,
) -> None:
    merged = encounter_medication_dataset.loc[:, DATASET_KEY_COLUMNS].merge(
        splits.loc[:, DATASET_KEY_COLUMNS],
        how="outer",
        on=DATASET_KEY_COLUMNS,
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        mismatch_count = int((merged["_merge"] != "both").sum())
        raise DataLoadError(
            "Dataset QC requires one-to-one alignment between the dataset and splits artifacts, "
            f"but found {mismatch_count:,} mismatched rows."
        )


def _assert_binary_column(
    dataframe: pd.DataFrame,
    column_name: str,
    *,
    artifact_name: str,
) -> None:
    values = pd.to_numeric(dataframe[column_name], errors="coerce")
    non_null_values = values[dataframe[column_name].notna()]
    if non_null_values.isna().any() or not set(non_null_values.astype(int).tolist()).issubset({0, 1}):
        raise DataLoadError(
            f"Artifact '{artifact_name}' has non-binary values in '{column_name}'."
        )


def _assert_non_null_column(
    dataframe: pd.DataFrame,
    column_name: str,
    *,
    artifact_name: str,
) -> None:
    if dataframe[column_name].isna().any():
        raise DataLoadError(
            f"Artifact '{artifact_name}' has null values in required column '{column_name}'."
        )


def _normalize_join_keys(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize analytical-grain key dtypes so parquet joins stay stable."""
    normalized = dataframe.copy()
    if "subject_id" in normalized.columns:
        normalized["subject_id"] = pd.to_numeric(
            normalized["subject_id"],
            errors="coerce",
        ).astype("Int64")
    for column_name in ("encounter_id", "medication_standardized"):
        if column_name in normalized.columns:
            normalized[column_name] = normalized[column_name].map(_normalize_text_key)
    if "review_timestamp" in normalized.columns:
        normalized["review_timestamp"] = pd.to_datetime(
            normalized["review_timestamp"],
            errors="coerce",
        )
    return normalized


def _normalize_text_key(value: object) -> str | None:
    """Normalize string-like analytical keys while preserving missing values."""
    if pd.isna(value):
        return None
    return str(value)
