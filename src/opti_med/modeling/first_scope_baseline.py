"""Baseline modeling and Phase 4 level-aware comparison scaffold for the 65+ first-scope hand-off."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from opti_med.data_access.exceptions import DataLoadError
from opti_med.evaluation.contracts import SplitPartition
from opti_med.modeling.first_scope_dataset import (
    DATASET_KEY_COLUMNS,
    select_benchmark_columns_from_dataset,
    select_trainable_feature_columns_from_dataset,
    validate_first_scope_dataset_artifact,
    validate_first_scope_split_artifact,
)
from opti_med.scoring.priority_levels import (
    PRIORITY_LEVELS,
    PRIORITY_SCORE_SEMANTIC_FAMILY,
    PriorityComparisonResult,
    build_priority_level_contract_payload,
    build_priority_comparison,
    derive_priority_level_series,
    resolve_priority_level_series,
)


FIRST_SCOPE_BASELINE_ARTIFACT_VERSION = "first_scope_baseline_v1_65plus_first_scope.v1"
PRIMARY_LABEL_COLUMN = "label__primary_action_binary"
ROW_ELIGIBILITY_COLUMN = "meta__dataset_row_eligible_for_training_flag"
SPLIT_COLUMN = "split_partition"
ROW_ID_COLUMN = "modeling__row_id"
AUDIT_COLUMNS = [
    ROW_ID_COLUMN,
    *DATASET_KEY_COLUMNS,
    "meta__hadm_id",
    "meta__stay_id",
    SPLIT_COLUMN,
    PRIMARY_LABEL_COLUMN,
    ROW_ELIGIBILITY_COLUMN,
]
DEFAULT_BENCHMARK_COMPARATORS: tuple[dict[str, str], ...] = (
    {
        "name": "benchmark_polypharmacy_only_flag",
        "column": "benchmark__polypharmacy_only_flag",
        "kind": "binary_flag",
        "description": "Polypharmacy-only hard-flag fallback.",
    },
    {
        "name": "benchmark_heuristic_any_supported_class_flag",
        "column": "benchmark__heuristic_any_supported_class_flag",
        "kind": "binary_flag",
        "description": "Any supported medication-class heuristic fallback.",
    },
    {
        "name": "benchmark_current_rule_score",
        "column": "benchmark__current_rule_score",
        "kind": "score",
        "description": "Legacy current-rule score comparison when populated.",
    },
)
SUSPICIOUS_FEATURE_TOKENS: tuple[str, ...] = (
    "benchmark__",
    "label__",
    "current_rule",
    "post_review",
)
LEVEL_COMPARISON_DETAIL_COLUMNS = [
    "comparison__left_source_name",
    "comparison__right_source_name",
    "comparison__left_score_value",
    "comparison__right_score_value",
    "comparison__left_level",
    "comparison__right_level",
    "comparison__status",
    "comparison__same_level_match_flag",
    "comparison__exact_score_match_flag",
    "comparison__absolute_score_error",
    "comparison__level_distance",
    "comparison__off_by_one_level_flag",
    "comparison__severe_disagreement_flag",
]
LEVEL_COMPARISON_SOURCE_SPECS: tuple[dict[str, str], ...] = (
    {
        "source_name": "prediction_priority_score",
        "score_column": "prediction__priority_score",
        "level_column": "prediction__priority_score_level",
        "semantic_family": PRIORITY_SCORE_SEMANTIC_FAMILY,
        "source_description": (
            "Future canonical 0-to-10 ML priority score. The current Phase 3 baseline only emits "
            "binary probabilities, not this score."
        ),
    },
    {
        "source_name": "benchmark_current_rule_score",
        "score_column": "benchmark__current_rule_score",
        "level_column": "benchmark__current_rule_score_level",
        "semantic_family": PRIORITY_SCORE_SEMANTIC_FAMILY,
        "source_description": "Rule-based benchmark score at the analytical grain when populated.",
    },
    {
        "source_name": "label_clinician_priority_score",
        "score_column": "label__clinician_priority_score",
        "level_column": "label__clinician_priority_score_level",
        "semantic_family": PRIORITY_SCORE_SEMANTIC_FAMILY,
        "source_description": (
            "Future clinician-approved 0-to-10 priority score. No such source is populated in the current workspace."
        ),
    },
)
LEVEL_COMPARISON_CANDIDATE_PAIRS: tuple[dict[str, str], ...] = (
    {
        "name": "prediction_priority_vs_benchmark_current_rule",
        "left_source_name": "prediction_priority_score",
        "right_source_name": "benchmark_current_rule_score",
    },
    {
        "name": "prediction_priority_vs_clinician_priority",
        "left_source_name": "prediction_priority_score",
        "right_source_name": "label_clinician_priority_score",
    },
    {
        "name": "benchmark_current_rule_vs_clinician_priority",
        "left_source_name": "benchmark_current_rule_score",
        "right_source_name": "label_clinician_priority_score",
    },
)


@dataclass(frozen=True, slots=True)
class FirstScopeBaselineSupervisedData:
    """Eligible supervised rows with features, labels, audit keys, and benchmarks."""

    rows: pd.DataFrame
    feature_columns: list[str]
    benchmark_columns: list[str]
    dataset_build_run_id: str
    split_build_run_id: str

    @property
    def y(self) -> pd.Series:
        """Return the canonical binary label."""
        return (
            pd.to_numeric(self.rows[PRIMARY_LABEL_COLUMN], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    @property
    def X(self) -> pd.DataFrame:
        """Return the raw trainable feature frame."""
        return self.rows.loc[:, self.feature_columns].copy()

    @property
    def benchmarks(self) -> pd.DataFrame:
        """Return the preserved benchmark-only columns."""
        return self.rows.loc[:, self.benchmark_columns].copy()


@dataclass(frozen=True, slots=True)
class FirstScopeBaselineRunResult:
    """Outputs from one deterministic baseline run."""

    run_id: str
    predictions: pd.DataFrame
    level_comparison_details: pd.DataFrame
    metrics: dict[str, Any]
    feature_manifest: dict[str, Any]
    report_markdown: str


def load_first_scope_baseline_inputs(
    *,
    dataset_path: Path,
    splits_path: Path,
    feature_list_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
    """Load the canonical Phase 2.5 dataset and split artifacts."""
    dataset = pd.read_parquet(dataset_path)
    splits = pd.read_parquet(splits_path)
    feature_list_payload: dict[str, Any] | None = None
    if feature_list_path is not None and feature_list_path.exists():
        feature_list_payload = json.loads(feature_list_path.read_text(encoding="utf-8"))
    return dataset, splits, feature_list_payload


def build_first_scope_baseline_supervised_data(
    *,
    encounter_medication_dataset: pd.DataFrame,
    splits: pd.DataFrame,
    feature_list_payload: dict[str, Any] | None = None,
) -> FirstScopeBaselineSupervisedData:
    """Build the canonical supervised subset and keep benchmarks separate from X."""
    validate_first_scope_dataset_artifact(encounter_medication_dataset)
    validate_first_scope_split_artifact(splits)
    _assert_split_alignment(
        encounter_medication_dataset=encounter_medication_dataset,
        splits=splits,
    )

    feature_columns = select_trainable_feature_columns_from_dataset(
        encounter_medication_dataset
    )
    benchmark_columns = select_benchmark_columns_from_dataset(
        encounter_medication_dataset
    )
    if not feature_columns:
        raise DataLoadError("Phase 3 baseline requires at least one feature__ column.")
    if set(feature_columns) & set(benchmark_columns):
        raise DataLoadError("benchmark__ columns leaked into the feature__ selection.")
    if feature_list_payload is not None:
        _validate_feature_list_contract(
            feature_list_payload=feature_list_payload,
            feature_columns=feature_columns,
            benchmark_columns=benchmark_columns,
        )

    merged = encounter_medication_dataset.merge(
        splits.loc[:, DATASET_KEY_COLUMNS + [SPLIT_COLUMN]],
        how="left",
        on=DATASET_KEY_COLUMNS,
        validate="one_to_one",
    )
    if merged[SPLIT_COLUMN].isna().any():
        raise DataLoadError(
            "Every dataset row must align to exactly one persisted split assignment."
        )

    eligible_mask = (
        pd.to_numeric(merged[ROW_ELIGIBILITY_COLUMN], errors="coerce")
        .fillna(0)
        .astype(int)
        .eq(1)
    )
    supervised = merged.loc[eligible_mask].copy()
    if supervised.empty:
        raise DataLoadError(
            "Phase 3 baseline requires at least one eligible supervised row."
        )

    duplicate_mask = supervised.duplicated(subset=DATASET_KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        raise DataLoadError(
            "Eligible supervised subset has duplicate analytical-grain rows."
        )

    supervised[ROW_ID_COLUMN] = supervised.apply(_build_row_id, axis=1)
    if supervised[ROW_ID_COLUMN].duplicated(keep=False).any():
        raise DataLoadError("Generated row identifiers must be unique.")

    _assert_no_infinite_feature_values(
        dataframe=supervised,
        feature_columns=feature_columns,
    )
    suspicious_feature_columns = sorted(
        column_name
        for column_name in feature_columns
        if any(token in column_name for token in SUSPICIOUS_FEATURE_TOKENS)
    )
    if suspicious_feature_columns:
        raise DataLoadError(
            "Phase 3 baseline refused suspicious feature__ columns that look label- or benchmark-derived: "
            + ", ".join(suspicious_feature_columns)
        )

    columns = [
        *AUDIT_COLUMNS,
        *feature_columns,
        *benchmark_columns,
    ]
    supervised = supervised.loc[:, columns].sort_values(
        DATASET_KEY_COLUMNS,
        na_position="last",
    )
    return FirstScopeBaselineSupervisedData(
        rows=supervised.reset_index(drop=True),
        feature_columns=feature_columns,
        benchmark_columns=benchmark_columns,
        dataset_build_run_id=_single_text_value(
            encounter_medication_dataset["meta__dataset_build_run_id"]
        ),
        split_build_run_id=_single_text_value(splits["split_build_run_id"]),
    )


def run_first_scope_baseline(
    *,
    supervised_data: FirstScopeBaselineSupervisedData,
    dataset_path: Path,
    splits_path: Path,
    feature_list_path: Path | None,
    random_seed: int,
) -> FirstScopeBaselineRunResult:
    """Fit the tiny baseline family, evaluate it safely, and render artifacts."""
    run_id = _stable_run_id(
        {
            "artifact_version": FIRST_SCOPE_BASELINE_ARTIFACT_VERSION,
            "dataset_build_run_id": supervised_data.dataset_build_run_id,
            "split_build_run_id": supervised_data.split_build_run_id,
            "random_seed": random_seed,
            "feature_columns": supervised_data.feature_columns,
        }
    )

    train_rows = supervised_data.rows.loc[
        supervised_data.rows[SPLIT_COLUMN] == SplitPartition.TRAIN.value
    ].copy()
    validation_rows = supervised_data.rows.loc[
        supervised_data.rows[SPLIT_COLUMN] == SplitPartition.VALIDATION.value
    ].copy()
    test_rows = supervised_data.rows.loc[
        supervised_data.rows[SPLIT_COLUMN] == SplitPartition.TEST.value
    ].copy()
    split_frames = {
        SplitPartition.TRAIN.value: train_rows,
        SplitPartition.VALIDATION.value: validation_rows,
        SplitPartition.TEST.value: test_rows,
    }
    train_y = (
        pd.to_numeric(train_rows[PRIMARY_LABEL_COLUMN], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    preprocessor_bundle = _fit_preprocessing_bundle(
        train_frame=train_rows.loc[:, supervised_data.feature_columns],
        feature_columns=supervised_data.feature_columns,
    )

    prediction_frames: list[pd.DataFrame] = []
    estimator_results: dict[str, Any] = {}
    estimators = _build_estimator_specs(random_seed=random_seed)
    for estimator_spec in estimators:
        estimator_payload = _fit_and_evaluate_estimator(
            estimator_spec=estimator_spec,
            supervised_data=supervised_data,
            train_y=train_y,
            split_frames=split_frames,
            preprocessor_bundle=preprocessor_bundle,
            run_id=run_id,
        )
        estimator_results[estimator_spec["name"]] = estimator_payload["metrics"]
        if estimator_payload["predictions"] is not None:
            prediction_frames.append(estimator_payload["predictions"])

    benchmark_results = _evaluate_benchmarks(
        rows=supervised_data.rows,
        benchmark_columns=supervised_data.benchmark_columns,
        split_frames=split_frames,
    )
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else _empty_prediction_artifact_frame(
            benchmark_columns=supervised_data.benchmark_columns
        )
    )
    if not predictions.empty:
        predictions = predictions.sort_values(
            ["estimator_name", SPLIT_COLUMN, ROW_ID_COLUMN],
            na_position="last",
        ).reset_index(drop=True)
    predictions = _add_level_aware_prediction_columns(predictions=predictions)
    level_aware_comparisons, level_comparison_details = (
        _build_level_aware_comparison_outputs(predictions=predictions)
    )

    feature_manifest = {
        "artifact_version": FIRST_SCOPE_BASELINE_ARTIFACT_VERSION,
        "run_id": run_id,
        "raw_feature_columns": supervised_data.feature_columns,
        "raw_feature_count": len(supervised_data.feature_columns),
        "numeric_feature_columns": preprocessor_bundle["numeric_columns"],
        "categorical_feature_columns": preprocessor_bundle["categorical_columns"],
        "dropped_all_missing_columns": preprocessor_bundle["dropped_all_missing_columns"],
        "dropped_constant_columns": preprocessor_bundle["dropped_constant_columns"],
        "retained_raw_feature_columns": preprocessor_bundle["retained_feature_columns"],
        "retained_raw_feature_count": len(preprocessor_bundle["retained_feature_columns"]),
        "transformed_feature_count": len(preprocessor_bundle["transformed_feature_names"]),
        "transformed_feature_names": preprocessor_bundle["transformed_feature_names"],
    }
    metrics = _build_run_metrics(
        supervised_data=supervised_data,
        run_id=run_id,
        dataset_path=dataset_path,
        splits_path=splits_path,
        feature_list_path=feature_list_path,
        random_seed=random_seed,
        feature_manifest=feature_manifest,
        estimator_results=estimator_results,
        benchmark_results=benchmark_results,
        level_aware_comparisons=level_aware_comparisons,
    )
    report_markdown = build_first_scope_baseline_report(metrics=metrics)
    return FirstScopeBaselineRunResult(
        run_id=run_id,
        predictions=predictions,
        level_comparison_details=level_comparison_details,
        metrics=metrics,
        feature_manifest=feature_manifest,
        report_markdown=report_markdown,
    )


def write_first_scope_baseline_artifacts(
    *,
    result: FirstScopeBaselineRunResult,
    output_root: Path,
    report_output: Path | None = None,
) -> dict[str, Path]:
    """Persist the canonical baseline and Phase 4 comparison artifacts."""
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": output_root / "predictions.parquet",
        "level_comparison_details": output_root / "level_aware_comparison_details.parquet",
        "metrics": output_root / "metrics.json",
        "feature_manifest": output_root / "feature_manifest.json",
        "report": output_root / "modeling_report.md",
    }
    result.predictions.to_parquet(paths["predictions"], index=False)
    result.level_comparison_details.to_parquet(
        paths["level_comparison_details"],
        index=False,
    )
    paths["metrics"].write_text(
        json.dumps(_json_ready(result.metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["feature_manifest"].write_text(
        json.dumps(_json_ready(result.feature_manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["report"].write_text(result.report_markdown, encoding="utf-8")
    if report_output is not None:
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(result.report_markdown, encoding="utf-8")
        paths["report_copy"] = report_output
    return paths


def build_first_scope_baseline_report(*, metrics: dict[str, Any]) -> str:
    """Render a compact QC-style markdown report for the baseline and Phase 4 comparison run."""
    lines: list[str] = [
        "# Phase 4 Level-Aware Baseline Modeling Report",
        "",
        "This run is a baseline plumbing milestone, not a performance milestone.",
        "Level agreement is now the primary clinical comparison lens whenever two sources share canonical 0-to-10 priority-score semantics.",
        "",
        "## Input Artifacts",
        "",
        f"- dataset: `{metrics['input_artifacts']['dataset_path']}`",
        f"- splits: `{metrics['input_artifacts']['splits_path']}`",
        f"- feature list: `{metrics['input_artifacts']['feature_list_path']}`",
        f"- dataset_build_run_id: `{metrics['dataset_build_run_id']}`",
        f"- split_build_run_id: `{metrics['split_build_run_id']}`",
        f"- baseline_run_id: `{metrics['run_id']}`",
        "",
        "## Training Contract",
        "",
        f"- canonical row filter: `{metrics['training_contract']['canonical_row_filter_expression']}`",
        f"- trainable inputs: `{metrics['training_contract']['trainable_feature_prefix']}*` only",
        f"- supervised target: `{metrics['training_contract']['label_column']}`",
        f"- benchmark columns excluded from model fitting: {metrics['training_contract']['benchmark_columns_excluded_from_training']}",
        f"- prediction row id column: `{metrics['auditability']['row_id_column']}`",
        "",
        "## Canonical Level Mapping",
        "",
        f"- semantic family: `{metrics['priority_level_contract']['semantic_family']}`",
        f"- score range: `{metrics['priority_level_contract']['score_min']:.0f}-{metrics['priority_level_contract']['score_max']:.0f}`",
        f"- ordered levels: `{', '.join(metrics['priority_level_contract']['level_order'])}`",
        f"- float bucketing policy: `{metrics['priority_level_contract']['float_bucketing_policy']}`",
        f"- primary clinical comparison lens: `{metrics['priority_level_contract']['primary_clinical_comparison_lens']}`",
        f"- exact numeric equality role: `{metrics['priority_level_contract']['exact_score_comparison_role']}`",
        "",
    ]
    for bucket in metrics["priority_level_contract"]["ordered_buckets"]:
        lines.append(f"- `{bucket['display_range']} => {bucket['level']}`")
    lines.extend(
        [
            "",
            "Level-based comparison is the primary lens here because the current supervised subset is too sparse for brittle exact-score claims to be clinically or statistically credible.",
            "",
            "## Supervised Subset",
            "",
            f"- eligible rows used for modeling: {metrics['eligible_row_count']:,}",
            f"- trainable feature count: {metrics['raw_feature_count']:,}",
            f"- retained raw feature count after train-only dropping: {metrics['retained_feature_count']:,}",
            f"- transformed feature count after one-hot expansion: {metrics['transformed_feature_count']:,}",
            "",
            "### Split Counts",
            "",
        ]
    )
    for split_name, payload in metrics["split_summary"].items():
        lines.append(
            f"- {split_name}: rows={payload['row_count']:,}, positives={payload['positive_count']:,}, "
            f"prevalence={payload['positive_prevalence']:.2%}"
        )
    lines.extend(
        [
            "",
            "## Preprocessing",
            "",
            "- numeric coercion is inferred from the training split only.",
            "- missing numeric values are median-imputed on train and carried to validation/test with the fitted imputers.",
            "- missing categorical values are filled with a deterministic `__missing__` sentinel and one-hot encoded with train-only categories.",
            f"- dropped all-missing train columns: {len(metrics['feature_manifest']['dropped_all_missing_columns']):,}",
            f"- dropped constant train columns: {len(metrics['feature_manifest']['dropped_constant_columns']):,}",
            "",
            "## Leakage Audit",
            "",
            "- dataset/split row alignment validated one-to-one before filtering.",
            "- preprocessing typing, imputers, and train-time column dropping are fit on train only.",
            "- validation/test rows are transformed only after train-time preprocessing is frozen.",
            "- feature names were audited for explicit label/current-rule/post-review leakage tokens before fitting.",
            "",
            "## Estimator Status",
            "",
        ]
    )
    for estimator_name, payload in metrics["estimators"].items():
        status = payload["status"]
        reason = payload["skip_reason"] or "fit"
        lines.append(f"- {estimator_name}: {status} ({reason})")
    lines.extend(
        [
            "",
            "## Estimator Metrics",
            "",
            "| estimator | split | rows | positives | confusion (tn/fp/fn/tp) | balanced_acc | precision | recall | f1 | pr_auc | roc_auc | brier | notes |",
            "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for estimator_name, payload in metrics["estimators"].items():
        if payload["status"] != "fit":
            continue
        for split_name in [
            SplitPartition.TRAIN.value,
            SplitPartition.VALIDATION.value,
            SplitPartition.TEST.value,
        ]:
            split_payload = payload["splits"][split_name]
            confusion = split_payload["confusion_matrix"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        estimator_name,
                        split_name,
                        f"{split_payload['row_count']:,}",
                        f"{split_payload['positive_count']:,}",
                        (
                            f"{confusion['tn']}/{confusion['fp']}/"
                            f"{confusion['fn']}/{confusion['tp']}"
                        ),
                        _format_metric_value(split_payload["metrics"]["balanced_accuracy"]),
                        _format_metric_value(split_payload["metrics"]["precision"]),
                        _format_metric_value(split_payload["metrics"]["recall"]),
                        _format_metric_value(split_payload["metrics"]["f1"]),
                        _format_metric_value(split_payload["metrics"]["pr_auc"]),
                        _format_metric_value(split_payload["metrics"]["roc_auc"]),
                        _format_metric_value(split_payload["metrics"]["brier_score"]),
                        _format_split_notes(split_payload["warnings"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Benchmark Comparison",
            "",
            "| benchmark | status | split | comparable rows | positives | confusion (tn/fp/fn/tp) | balanced_acc | precision | recall | f1 | pr_auc | roc_auc | brier | notes |",
            "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for benchmark_name, payload in metrics["benchmarks"].items():
        if payload["status"] != "available":
            lines.append(
                "| "
                + " | ".join(
                    [
                        benchmark_name,
                        payload["status"],
                        "n/a",
                        "0",
                        "0",
                        "n/a",
                        "n/a",
                        "n/a",
                        "n/a",
                        "n/a",
                        "n/a",
                        "n/a",
                        "n/a",
                        payload.get("status_reason", "not available"),
                    ]
                )
                + " |"
            )
            continue
        for split_name in [
            SplitPartition.TRAIN.value,
            SplitPartition.VALIDATION.value,
            SplitPartition.TEST.value,
        ]:
            split_payload = payload["splits"][split_name]
            confusion = split_payload["confusion_matrix"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        benchmark_name,
                        payload["status"],
                        split_name,
                        f"{split_payload['row_count']:,}",
                        f"{split_payload['positive_count']:,}",
                        (
                            f"{confusion['tn']}/{confusion['fp']}/"
                            f"{confusion['fn']}/{confusion['tp']}"
                        ),
                        _format_metric_value(split_payload["metrics"]["balanced_accuracy"]),
                        _format_metric_value(split_payload["metrics"]["precision"]),
                        _format_metric_value(split_payload["metrics"]["recall"]),
                        _format_metric_value(split_payload["metrics"]["f1"]),
                        _format_metric_value(split_payload["metrics"]["pr_auc"]),
                        _format_metric_value(split_payload["metrics"]["roc_auc"]),
                        _format_metric_value(split_payload["metrics"]["brier_score"]),
                        _format_split_notes(split_payload["warnings"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Level-Aware Comparison Availability",
            "",
            "- Only sources declared under the canonical 0-to-10 priority-score semantic family are eligible for ordinal comparison.",
        ]
    )
    for source_name, payload in metrics["level_aware_comparisons"]["source_catalog"].items():
        lines.append(
            f"- {source_name}: score_column_present={payload['score_column_present']}, "
            f"level_column_present={payload['level_column_present']}, "
            f"semantic_family={payload['semantic_family']}"
        )
    lines.extend(
        [
            "",
            "## Level-Aware Comparison Summary",
            "",
            "| estimator | candidate | status | level rows | score rows | same_level_agreement | exact_score_agreement | mean_abs_score_error | off_by_one | severe_disagreement | weighted_kappa | notes |",
            "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    pairwise = metrics["level_aware_comparisons"]["pairwise"]
    for estimator_name in sorted(pairwise):
        for candidate_name, payload in pairwise[estimator_name].items():
            if payload["status"] != "available":
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            estimator_name,
                            candidate_name,
                            payload["status"],
                            "0",
                            "0",
                            "n/a",
                            "n/a",
                            "n/a",
                            "n/a",
                            "n/a",
                            "n/a",
                            payload["status_reason"],
                        ]
                    )
                    + " |"
                )
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        estimator_name,
                        candidate_name,
                        payload["status"],
                        f"{payload['level_comparable_row_count']:,}",
                        f"{payload['score_comparable_row_count']:,}",
                        _format_optional_count_rate(
                            payload["same_level_agreement_count"],
                            payload["same_level_agreement_rate"],
                        ),
                        _format_optional_count_rate(
                            payload["exact_score_agreement_count"],
                            payload["exact_score_agreement_rate"],
                        ),
                        _format_optional_metric(payload["mean_absolute_score_error"]),
                        _format_optional_count_rate(
                            payload["off_by_one_level_count"],
                            payload["off_by_one_level_rate"],
                        ),
                        _format_optional_count_rate(
                            payload["severe_disagreement_count"],
                            payload["severe_disagreement_rate"],
                        ),
                        _format_optional_metric(payload["weighted_kappa_quadratic"]),
                        "; ".join(payload["warnings"]) if payload["warnings"] else "none",
                    ]
                )
                + " |"
            )
            lines.append("")
            lines.append(f"Ordinal confusion for `{estimator_name}` / `{candidate_name}`:")
            for left_level in PRIORITY_LEVELS:
                row = payload["ordinal_confusion_matrix"][left_level]
                lines.append(
                    f"- {left_level}: low={row['low']}, medium={row['medium']}, high={row['high']}"
                )
    lines.extend(
        [
            "",
            "## Warnings",
            "",
        ]
    )
    for warning in metrics["warnings"]:
        lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def _build_run_metrics(
    *,
    supervised_data: FirstScopeBaselineSupervisedData,
    run_id: str,
    dataset_path: Path,
    splits_path: Path,
    feature_list_path: Path | None,
    random_seed: int,
    feature_manifest: dict[str, Any],
    estimator_results: dict[str, Any],
    benchmark_results: dict[str, Any],
    level_aware_comparisons: dict[str, Any],
) -> dict[str, Any]:
    """Build the nested metrics payload for JSON and report generation."""
    split_summary: dict[str, Any] = {}
    warnings: set[str] = set()
    for split_name in [
        SplitPartition.TRAIN.value,
        SplitPartition.VALIDATION.value,
        SplitPartition.TEST.value,
    ]:
        split_rows = supervised_data.rows.loc[
            supervised_data.rows[SPLIT_COLUMN] == split_name
        ]
        split_y = pd.to_numeric(split_rows[PRIMARY_LABEL_COLUMN], errors="coerce").fillna(0).astype(int)
        positive_count = int(split_y.eq(1).sum())
        split_summary[split_name] = {
            "row_count": int(len(split_rows)),
            "positive_count": positive_count,
            "positive_prevalence": (
                positive_count / int(len(split_rows)) if len(split_rows) else 0.0
            ),
        }
        if positive_count == 0:
            warnings.add(
                f"{split_name} split has zero positive labels after applying the canonical training filter."
            )
    if (
        split_summary[SplitPartition.VALIDATION.value]["positive_count"] == 0
        and split_summary[SplitPartition.TEST.value]["positive_count"] == 0
    ):
        warnings.add(
            "Held-out validation and test partitions contain zero positive labels, so positive-class metrics and rank metrics are not computable there."
        )
        warnings.add(
            "Any train-only separation in this run is not evidence of generalization because no held-out positive cases exist."
        )
    if int(supervised_data.y.eq(1).sum()) <= 2:
        warnings.add(
            "Only two positive eligible rows exist overall, so all probabilistic outputs should be treated as unstable and non-calibration-grade."
        )
    for warning in level_aware_comparisons.get("warnings", []):
        warnings.add(str(warning))
    return {
        "artifact_version": FIRST_SCOPE_BASELINE_ARTIFACT_VERSION,
        "run_id": run_id,
        "random_seed": random_seed,
        "input_artifacts": {
            "dataset_path": str(dataset_path),
            "splits_path": str(splits_path),
            "feature_list_path": str(feature_list_path) if feature_list_path is not None else None,
        },
        "training_contract": {
            "canonical_row_filter_expression": f"{ROW_ELIGIBILITY_COLUMN} == 1",
            "trainable_feature_prefix": "feature__",
            "label_column": PRIMARY_LABEL_COLUMN,
            "benchmark_prefix": "benchmark__",
            "benchmark_columns_excluded_from_training": True,
        },
        "auditability": {
            "row_id_column": ROW_ID_COLUMN,
            "primary_key_columns": list(DATASET_KEY_COLUMNS),
            "prediction_artifact_contains_row_identifiers": True,
        },
        "leakage_audit": {
            "split_alignment_validated": True,
            "preprocessing_fit_on_train_only": True,
            "validation_and_test_excluded_from_fit": True,
            "suspicious_feature_tokens_checked": list(SUSPICIOUS_FEATURE_TOKENS),
            "suspicious_feature_columns_found": [],
        },
        "dataset_build_run_id": supervised_data.dataset_build_run_id,
        "split_build_run_id": supervised_data.split_build_run_id,
        "eligible_row_count": int(len(supervised_data.rows)),
        "raw_feature_count": len(supervised_data.feature_columns),
        "retained_feature_count": feature_manifest["retained_raw_feature_count"],
        "transformed_feature_count": feature_manifest["transformed_feature_count"],
        "split_summary": split_summary,
        "feature_manifest": feature_manifest,
        "estimators": estimator_results,
        "benchmarks": benchmark_results,
        "priority_level_contract": build_priority_level_contract_payload(),
        "level_aware_comparisons": level_aware_comparisons,
        "warnings": sorted(warnings),
    }


def _fit_and_evaluate_estimator(
    *,
    estimator_spec: dict[str, Any],
    supervised_data: FirstScopeBaselineSupervisedData,
    train_y: pd.Series,
    split_frames: dict[str, pd.DataFrame],
    preprocessor_bundle: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Fit one estimator when viable and evaluate it across splits."""
    skip_reason = _validate_estimator_viability(
        estimator_name=estimator_spec["name"],
        train_y=train_y,
        retained_feature_columns=preprocessor_bundle["retained_feature_columns"],
    )
    if skip_reason is not None:
        return {
            "metrics": {
                "status": "skipped",
                "family": estimator_spec["family"],
                "skip_reason": skip_reason,
                "splits": _empty_split_metrics(),
            },
            "predictions": None,
        }

    estimator = estimator_spec["factory"](preprocessor_bundle["preprocessor"])
    train_X = (
        _dummy_feature_frame(split_frames[SplitPartition.TRAIN.value])
        if estimator_spec["family"] == "dummy"
        else split_frames[SplitPartition.TRAIN.value].loc[:, supervised_data.feature_columns]
    )
    estimator.fit(
        train_X,
        train_y.to_numpy(dtype=int),
    )

    split_metrics: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []
    for split_name, split_rows in split_frames.items():
        y_true = (
            pd.to_numeric(split_rows[PRIMARY_LABEL_COLUMN], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        if split_rows.empty:
            split_metrics[split_name] = _evaluate_binary_predictions(
                y_true=np.array([], dtype=int),
                y_pred=np.array([], dtype=int),
                y_score=np.array([], dtype=float),
            )
            continue
        split_X = (
            _dummy_feature_frame(split_rows)
            if estimator_spec["family"] == "dummy"
            else split_rows.loc[:, supervised_data.feature_columns]
        )
        predicted_label = estimator.predict(
            split_X
        ).astype(int)
        if hasattr(estimator, "predict_proba"):
            predicted_probability = _predict_probability_of_positive(
                estimator=estimator,
                features=split_X,
            )
        else:
            predicted_probability = predicted_label.astype(float)
        split_metrics[split_name] = _evaluate_binary_predictions(
            y_true=y_true.to_numpy(dtype=int),
            y_pred=np.asarray(predicted_label, dtype=int),
            y_score=np.asarray(predicted_probability, dtype=float),
        )
        prediction_frames.append(
            _build_prediction_frame(
                split_rows=split_rows,
                benchmark_columns=supervised_data.benchmark_columns,
                estimator_name=estimator_spec["name"],
                estimator_family=estimator_spec["family"],
                run_id=run_id,
                predicted_label=np.asarray(predicted_label, dtype=int),
                predicted_probability=np.asarray(predicted_probability, dtype=float),
            )
        )

    return {
        "metrics": {
            "status": "fit",
            "family": estimator_spec["family"],
            "skip_reason": None,
            "splits": split_metrics,
        },
        "predictions": pd.concat(prediction_frames, ignore_index=True),
    }


def _evaluate_benchmarks(
    *,
    rows: pd.DataFrame,
    benchmark_columns: list[str],
    split_frames: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Evaluate the supported benchmark-only comparison columns."""
    results: dict[str, Any] = {}
    for spec in DEFAULT_BENCHMARK_COMPARATORS:
        column_name = spec["column"]
        comparator_name = spec["name"]
        if column_name not in benchmark_columns:
            results[comparator_name] = {
                "status": "missing_column",
                "status_reason": f"{column_name} is absent from the eligible modeling dataset.",
                "description": spec["description"],
                "column_name": column_name,
                "splits": _empty_split_metrics(),
            }
            continue
        non_null_count = int(rows[column_name].notna().sum())
        if non_null_count == 0:
            results[comparator_name] = {
                "status": "unpopulated",
                "status_reason": (
                    f"{column_name} is present but unpopulated on the eligible modeling subset; "
                    "no valid benchmark comparison is available."
                ),
                "description": spec["description"],
                "column_name": column_name,
                "splits": _empty_split_metrics(),
            }
            continue
        split_metrics: dict[str, Any] = {}
        for split_name, split_rows in split_frames.items():
            evaluation = _evaluate_benchmark_split(
                split_rows=split_rows,
                spec=spec,
            )
            split_metrics[split_name] = evaluation
        results[comparator_name] = {
            "status": "available",
            "status_reason": f"{column_name} is populated on the eligible modeling subset.",
            "description": spec["description"],
            "column_name": column_name,
            "splits": split_metrics,
        }
    return results


def _evaluate_benchmark_split(
    *,
    split_rows: pd.DataFrame,
    spec: dict[str, str],
) -> dict[str, Any]:
    """Evaluate one benchmark comparator on one split."""
    if split_rows.empty:
        return _evaluate_binary_predictions(
            y_true=np.array([], dtype=int),
            y_pred=np.array([], dtype=int),
            y_score=np.array([], dtype=float),
        )
    column_name = spec["column"]
    y_true = (
        pd.to_numeric(split_rows[PRIMARY_LABEL_COLUMN], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    values = split_rows[column_name]
    if spec["kind"] == "binary_flag":
        score = pd.to_numeric(values, errors="coerce")
        available_mask = score.notna()
        filtered_y = y_true.loc[available_mask].to_numpy(dtype=int)
        filtered_score = score.loc[available_mask].astype(float).to_numpy()
        filtered_pred = score.loc[available_mask].fillna(0).astype(int).to_numpy()
    else:
        score = pd.to_numeric(values, errors="coerce")
        available_mask = score.notna()
        filtered_y = y_true.loc[available_mask].to_numpy(dtype=int)
        filtered_score = score.loc[available_mask].astype(float).to_numpy()
        filtered_pred = (filtered_score >= 0.5).astype(int)
    return _evaluate_binary_predictions(
        y_true=filtered_y,
        y_pred=filtered_pred,
        y_score=filtered_score,
    )


def _fit_preprocessing_bundle(
    *,
    train_frame: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    """Fit deterministic train-only preprocessing for the baseline pipeline."""
    numeric_columns, categorical_columns = _infer_feature_types(
        dataframe=train_frame,
        feature_columns=feature_columns,
    )
    coerced_train = _coerce_feature_frame(
        dataframe=train_frame,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    dropped_all_missing_columns = sorted(
        column_name
        for column_name in feature_columns
        if _is_all_missing(coerced_train[column_name])
    )
    retained_feature_columns = [
        column_name
        for column_name in feature_columns
        if column_name not in dropped_all_missing_columns
    ]
    dropped_constant_columns = sorted(
        column_name
        for column_name in retained_feature_columns
        if _is_constant_after_coercion(coerced_train[column_name])
    )
    retained_feature_columns = [
        column_name
        for column_name in retained_feature_columns
        if column_name not in dropped_constant_columns
    ]
    retained_numeric_columns = [
        column_name
        for column_name in numeric_columns
        if column_name in retained_feature_columns
    ]
    retained_categorical_columns = [
        column_name
        for column_name in categorical_columns
        if column_name in retained_feature_columns
    ]
    if retained_numeric_columns or retained_categorical_columns:
        transformers = []
        if retained_numeric_columns:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            (
                                "coerce_numeric",
                                FunctionTransformer(
                                    _coerce_numeric_frame_for_sklearn,
                                    validate=False,
                                    feature_names_out="one-to-one",
                                ),
                            ),
                            ("imputer", SimpleImputer(strategy="median")),
                        ]
                    ),
                    retained_numeric_columns,
                )
            )
        if retained_categorical_columns:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            (
                                "coerce_categorical",
                                FunctionTransformer(
                                    _coerce_categorical_frame_for_sklearn,
                                    validate=False,
                                    feature_names_out="one-to-one",
                                ),
                            ),
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="constant",
                                    fill_value="__missing__",
                                ),
                            ),
                            (
                                "encoder",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=False,
                                ),
                            ),
                        ]
                    ),
                    retained_categorical_columns,
                )
            )
        preprocessor = ColumnTransformer(transformers=transformers)
        preprocessor.fit(train_frame.loc[:, retained_feature_columns])
        transformed_feature_names = preprocessor.get_feature_names_out().tolist()
    else:
        preprocessor = None
        transformed_feature_names = []
    return {
        "preprocessor": preprocessor,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "retained_feature_columns": retained_feature_columns,
        "dropped_all_missing_columns": dropped_all_missing_columns,
        "dropped_constant_columns": dropped_constant_columns,
        "transformed_feature_names": transformed_feature_names,
    }


def _build_estimator_specs(*, random_seed: int) -> list[dict[str, Any]]:
    """Return the intentionally small baseline family for the current scaffold."""
    return [
        {
            "name": "dummy_majority",
            "family": "dummy",
            "factory": lambda _preprocessor: DummyClassifier(strategy="prior"),
        },
        {
            "name": "dummy_stratified",
            "family": "dummy",
            "factory": lambda _preprocessor: DummyClassifier(
                strategy="stratified",
                random_state=random_seed,
            ),
        },
        {
            "name": "logistic_regression_l2",
            "family": "linear_probabilistic",
            "factory": lambda preprocessor: Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    (
                        "model",
                        LogisticRegression(
                            C=0.1,
                            solver="liblinear",
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=random_seed,
                        ),
                    ),
                ]
            ),
        },
    ]


def _validate_estimator_viability(
    *,
    estimator_name: str,
    train_y: pd.Series,
    retained_feature_columns: list[str],
) -> str | None:
    """Return a skip reason when the estimator is not viable on the train split."""
    if train_y.empty:
        return "training split has no eligible supervised rows"
    class_counts = train_y.value_counts().sort_index()
    unique_class_count = int(class_counts.index.nunique())
    if estimator_name == "dummy_stratified" and unique_class_count < 2:
        return "stratified dummy requires both classes in the training split"
    if estimator_name == "logistic_regression_l2":
        if unique_class_count < 2:
            return "logistic regression requires both classes in the training split"
        if int(class_counts.min()) < 3:
            return (
                "logistic regression requires at least three training rows in the minority class for a minimally stable baseline"
            )
        if not retained_feature_columns:
            return "all trainable features were dropped during train-only preprocessing"
    return None


def _build_prediction_frame(
    *,
    split_rows: pd.DataFrame,
    benchmark_columns: list[str],
    estimator_name: str,
    estimator_family: str,
    run_id: str,
    predicted_label: np.ndarray,
    predicted_probability: np.ndarray,
) -> pd.DataFrame:
    """Build an auditable prediction artifact frame for one estimator/split."""
    output = split_rows.loc[:, [*AUDIT_COLUMNS, *benchmark_columns]].copy()
    output["artifact_version"] = FIRST_SCOPE_BASELINE_ARTIFACT_VERSION
    output["model_run_id"] = run_id
    output["estimator_name"] = estimator_name
    output["estimator_family"] = estimator_family
    output["predicted_label"] = predicted_label.astype(int)
    output["predicted_probability"] = predicted_probability.astype(float)
    output["predicted_score"] = predicted_probability.astype(float)
    ordered = [
        "artifact_version",
        "model_run_id",
        "estimator_name",
        "estimator_family",
        *AUDIT_COLUMNS,
        "predicted_label",
        "predicted_probability",
        "predicted_score",
        *benchmark_columns,
    ]
    return output.loc[:, ordered]


def _dummy_feature_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a deterministic placeholder feature frame for dummy baselines."""
    return pd.DataFrame({"__dummy__": np.zeros(len(dataframe), dtype=float)})


def _empty_prediction_artifact_frame(*, benchmark_columns: list[str]) -> pd.DataFrame:
    """Return a stable empty prediction artifact schema for all-skipped runs."""
    ordered = [
        "artifact_version",
        "model_run_id",
        "estimator_name",
        "estimator_family",
        *AUDIT_COLUMNS,
        "predicted_label",
        "predicted_probability",
        "predicted_score",
        *benchmark_columns,
    ]
    return pd.DataFrame(columns=ordered)


def _add_level_aware_prediction_columns(*, predictions: pd.DataFrame) -> pd.DataFrame:
    """Add derived level columns for score sources that already have canonical 0-to-10 semantics."""
    if predictions.empty:
        return predictions
    output = predictions.copy()
    if "benchmark__current_rule_score" in output.columns:
        output["benchmark__current_rule_score_level"] = derive_priority_level_series(
            output["benchmark__current_rule_score"]
        )
    return output


def _build_level_aware_comparison_outputs(
    *,
    predictions: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build pairwise level-aware comparison summaries and rowwise details when possible."""
    source_catalog = {
        spec["source_name"]: {
            "score_column": spec["score_column"],
            "level_column": spec["level_column"],
            "semantic_family": spec["semantic_family"],
            "source_description": spec["source_description"],
            "score_column_present": spec["score_column"] in predictions.columns,
            "level_column_present": spec["level_column"] in predictions.columns,
        }
        for spec in LEVEL_COMPARISON_SOURCE_SPECS
    }
    if predictions.empty:
        return (
            {
                "source_catalog": source_catalog,
                "pairwise": {},
                "warnings": ["Prediction artifact contains no rows."],
            },
            _empty_level_comparison_details(),
        )

    pairwise: dict[str, Any] = {}
    detail_frames: list[pd.DataFrame] = []
    key_columns = [
        "artifact_version",
        "model_run_id",
        "estimator_name",
        "estimator_family",
        *AUDIT_COLUMNS,
    ]
    for estimator_name, estimator_rows in predictions.groupby("estimator_name", sort=True):
        estimator_rows = estimator_rows.sort_values(
            [SPLIT_COLUMN, ROW_ID_COLUMN],
            na_position="last",
        ).reset_index(drop=True)
        estimator_pairwise: dict[str, Any] = {}
        for pair_spec in LEVEL_COMPARISON_CANDIDATE_PAIRS:
            left_spec = source_catalog[pair_spec["left_source_name"]]
            right_spec = source_catalog[pair_spec["right_source_name"]]
            availability = _resolve_level_comparison_availability(
                estimator_rows=estimator_rows,
                left_source_name=pair_spec["left_source_name"],
                right_source_name=pair_spec["right_source_name"],
                left_spec=left_spec,
                right_spec=right_spec,
            )
            if availability["status"] != "available":
                estimator_pairwise[pair_spec["name"]] = availability
                continue
            comparison_result = build_priority_comparison(
                key_frame=estimator_rows.loc[:, key_columns],
                left_source_name=pair_spec["left_source_name"],
                right_source_name=pair_spec["right_source_name"],
                left_score_series=(
                    estimator_rows[left_spec["score_column"]]
                    if left_spec["score_column_present"]
                    else None
                ),
                right_score_series=(
                    estimator_rows[right_spec["score_column"]]
                    if right_spec["score_column_present"]
                    else None
                ),
                left_level_series=(
                    estimator_rows[left_spec["level_column"]]
                    if left_spec["level_column_present"]
                    else None
                ),
                right_level_series=(
                    estimator_rows[right_spec["level_column"]]
                    if right_spec["level_column_present"]
                    else None
                ),
            )
            detail_frame = comparison_result.rowwise.copy()
            detail_frame["comparison__candidate_name"] = pair_spec["name"]
            detail_frames.append(detail_frame)
            estimator_pairwise[pair_spec["name"]] = comparison_result.summary
        pairwise[estimator_name] = estimator_pairwise
    details = (
        pd.concat(detail_frames, ignore_index=True)
        if detail_frames
        else _empty_level_comparison_details()
    )
    if not details.empty:
        details = details.sort_values(
            ["estimator_name", "comparison__candidate_name", SPLIT_COLUMN, ROW_ID_COLUMN],
            na_position="last",
        ).reset_index(drop=True)
    warnings = []
    if not detail_frames:
        warnings.append(
            "No current prediction artifact exposes a canonical 0-to-10 ML priority score, and no populated clinician-priority score source exists in the current workspace."
        )
    return {
        "source_catalog": source_catalog,
        "pairwise": pairwise,
        "warnings": warnings,
    }, details


def _resolve_level_comparison_availability(
    *,
    estimator_rows: pd.DataFrame,
    left_source_name: str,
    right_source_name: str,
    left_spec: dict[str, Any],
    right_spec: dict[str, Any],
) -> dict[str, Any]:
    """Return availability status for one candidate level-aware comparison pair."""
    if left_spec["semantic_family"] != right_spec["semantic_family"]:
        return {
            "status": "unavailable",
            "status_reason": (
                f"{left_source_name} and {right_source_name} do not share the same ordinal score semantic family."
            ),
        }
    if not left_spec["score_column_present"] and not left_spec["level_column_present"]:
        return {
            "status": "unavailable",
            "status_reason": (
                f"{left_source_name} is not present in the current artifact. "
                f"{left_spec['source_description']}"
            ),
        }
    if not right_spec["score_column_present"] and not right_spec["level_column_present"]:
        return {
            "status": "unavailable",
            "status_reason": (
                f"{right_source_name} is not present in the current artifact. "
                f"{right_spec['source_description']}"
            ),
        }
    left_level_series = resolve_priority_level_series(
        score_series=(
            estimator_rows[left_spec["score_column"]]
            if left_spec["score_column_present"]
            else None
        ),
        level_series=(
            estimator_rows[left_spec["level_column"]]
            if left_spec["level_column_present"]
            else None
        ),
        score_name=left_source_name,
        level_name=left_spec["level_column"],
    )
    right_level_series = resolve_priority_level_series(
        score_series=(
            estimator_rows[right_spec["score_column"]]
            if right_spec["score_column_present"]
            else None
        ),
        level_series=(
            estimator_rows[right_spec["level_column"]]
            if right_spec["level_column_present"]
            else None
        ),
        score_name=right_source_name,
        level_name=right_spec["level_column"],
    )
    if int(left_level_series.notna().sum()) == 0:
        return {
            "status": "unavailable",
            "status_reason": f"{left_source_name} has no populated ordinal level values in this artifact.",
        }
    if int(right_level_series.notna().sum()) == 0:
        return {
            "status": "unavailable",
            "status_reason": f"{right_source_name} has no populated ordinal level values in this artifact.",
        }
    return {"status": "available", "status_reason": "comparison is available"}


def _empty_level_comparison_details() -> pd.DataFrame:
    """Return a stable empty schema for rowwise level-aware comparisons."""
    columns = [
        "artifact_version",
        "model_run_id",
        "estimator_name",
        "estimator_family",
        *AUDIT_COLUMNS,
        *LEVEL_COMPARISON_DETAIL_COLUMNS,
        "comparison__candidate_name",
    ]
    return pd.DataFrame(columns=columns)


def _predict_probability_of_positive(
    *,
    estimator: Any,
    features: pd.DataFrame,
) -> np.ndarray:
    """Return the class-1 probability even when the estimator saw one class only."""
    probabilities = estimator.predict_proba(features)
    classes = getattr(estimator, "classes_", None)
    if classes is None and hasattr(estimator, "named_steps"):
        classes = getattr(estimator.named_steps.get("model"), "classes_", None)
    if classes is None:
        raise ValueError("Estimator does not expose classes_ for probability alignment.")
    classes_array = np.asarray(classes)
    if 1 in classes_array:
        positive_index = int(np.where(classes_array == 1)[0][0])
        return probabilities[:, positive_index].astype(float)
    if 0 in classes_array:
        return np.zeros(len(features), dtype=float)
    return np.full(len(features), np.nan, dtype=float)


def _evaluate_binary_predictions(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
) -> dict[str, Any]:
    """Evaluate binary predictions without overstating unsupported metrics."""
    row_count = int(len(y_true))
    if row_count == 0:
        return {
            "row_count": 0,
            "positive_count": 0,
            "positive_prevalence": 0.0,
            "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
            "metrics": _empty_metric_bundle("no_rows_available"),
            "warnings": ["No rows available for this split."],
        }

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score_array = None if y_score is None else np.asarray(y_score, dtype=float)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    predicted_positives = int((y_pred == 1).sum())
    warnings: list[str] = []

    metrics = _empty_metric_bundle()
    if positives > 0 and negatives > 0:
        recall_positive = tp / (tp + fn)
        recall_negative = tn / (tn + fp)
        metrics["balanced_accuracy"] = _ok_metric(
            (recall_positive + recall_negative) / 2.0
        )
    else:
        metrics["balanced_accuracy"] = _na_metric(
            "requires both positive and negative ground-truth rows"
        )
        warnings.append("Ground truth contains only one class; balanced accuracy skipped.")

    if predicted_positives > 0:
        metrics["precision"] = _ok_metric(tp / predicted_positives)
    else:
        metrics["precision"] = _na_metric("no predicted positive rows")
        warnings.append("No predicted positives; precision skipped.")

    if positives > 0:
        recall_value = tp / (tp + fn)
        metrics["recall"] = _ok_metric(recall_value)
    else:
        metrics["recall"] = _na_metric("no positive ground-truth rows")
        warnings.append("No positive ground truth; recall skipped.")

    if (
        metrics["precision"]["status"] == "ok"
        and metrics["recall"]["status"] == "ok"
    ):
        precision_value = float(metrics["precision"]["value"])
        recall_value = float(metrics["recall"]["value"])
        if precision_value + recall_value == 0:
            metrics["f1"] = _ok_metric(0.0)
        else:
            metrics["f1"] = _ok_metric(
                2.0 * precision_value * recall_value / (precision_value + recall_value)
            )
    else:
        metrics["f1"] = _na_metric("requires computable precision and recall")

    if y_score_array is not None and np.isfinite(y_score_array).all():
        if positives > 0 and negatives > 0:
            metrics["pr_auc"] = _ok_metric(
                float(average_precision_score(y_true, y_score_array))
            )
            metrics["roc_auc"] = _ok_metric(float(roc_auc_score(y_true, y_score_array)))
        else:
            metrics["pr_auc"] = _na_metric(
                "requires both positive and negative ground-truth rows"
            )
            metrics["roc_auc"] = _na_metric(
                "requires both positive and negative ground-truth rows"
            )
        metrics["brier_score"] = _ok_metric(
            float(np.mean((y_score_array - y_true.astype(float)) ** 2))
        )
        if positives < 10 or row_count < 25:
            warnings.append(
                "Probability estimates are not calibration-grade at this split size; treat Brier score as a rough diagnostic only."
            )
    else:
        metrics["pr_auc"] = _na_metric("no finite score output available")
        metrics["roc_auc"] = _na_metric("no finite score output available")
        metrics["brier_score"] = _na_metric("no finite score output available")

    if positives < 5:
        warnings.append(
            "Positive support is extremely small; all split-level metrics are high-variance."
        )
    return {
        "row_count": row_count,
        "positive_count": positives,
        "positive_prevalence": positives / row_count if row_count else 0.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "metrics": metrics,
        "warnings": sorted(set(warnings)),
    }


def _empty_split_metrics() -> dict[str, Any]:
    """Return a stable split-payload template for skipped estimators or benchmarks."""
    return {
        split_name: {
            "row_count": 0,
            "positive_count": 0,
            "positive_prevalence": 0.0,
            "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
            "metrics": _empty_metric_bundle("not_run"),
            "warnings": [],
        }
        for split_name in [
            SplitPartition.TRAIN.value,
            SplitPartition.VALIDATION.value,
            SplitPartition.TEST.value,
        ]
    }


def _empty_metric_bundle(reason: str | None = None) -> dict[str, Any]:
    """Return all requested metrics with explicit non-computable payloads."""
    return {
        "balanced_accuracy": _na_metric(reason or "not_computed"),
        "precision": _na_metric(reason or "not_computed"),
        "recall": _na_metric(reason or "not_computed"),
        "f1": _na_metric(reason or "not_computed"),
        "pr_auc": _na_metric(reason or "not_computed"),
        "roc_auc": _na_metric(reason or "not_computed"),
        "brier_score": _na_metric(reason or "not_computed"),
    }


def _ok_metric(value: float) -> dict[str, Any]:
    """Wrap a valid metric value."""
    return {"status": "ok", "value": float(value), "reason": None}


def _na_metric(reason: str) -> dict[str, Any]:
    """Wrap a skipped or non-computable metric."""
    return {"status": "not_computable", "value": None, "reason": reason}


def _format_metric_value(metric_payload: dict[str, Any]) -> str:
    """Format one metric payload for markdown tables."""
    if metric_payload["status"] != "ok":
        return f"n/a ({metric_payload['reason']})"
    return f"{float(metric_payload['value']):.4f}"


def _format_split_notes(warnings: list[str]) -> str:
    """Render split-level reliability notes compactly for markdown tables."""
    if not warnings:
        return "none"
    return "; ".join(sorted(set(warnings)))


def _format_optional_metric(value: float | None) -> str:
    """Format an optional scalar metric for markdown."""
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _format_optional_count_rate(count: int, rate: float | None) -> str:
    """Format count-plus-rate diagnostics for markdown tables."""
    if rate is None:
        return "n/a"
    return f"{count:,} ({float(rate):.4f})"


def _infer_feature_types(
    *,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    """Infer numeric vs categorical features from the training split only."""
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column_name in feature_columns:
        series = dataframe[column_name]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric_columns.append(column_name)
            continue
        non_null_mask = series.notna()
        coerced = pd.to_numeric(series, errors="coerce")
        if non_null_mask.any() and coerced.loc[non_null_mask].notna().all():
            numeric_columns.append(column_name)
        else:
            categorical_columns.append(column_name)
    return numeric_columns, categorical_columns


def _coerce_feature_frame(
    *,
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Apply deterministic numeric/categorical coercion."""
    output = dataframe.copy()
    for column_name in numeric_columns:
        output[column_name] = pd.to_numeric(output[column_name], errors="coerce")
    for column_name in categorical_columns:
        series = output[column_name].astype(object)
        output[column_name] = series.mask(pd.isna(series), np.nan)
    return output


def _coerce_numeric_frame_for_sklearn(dataframe: Any) -> pd.DataFrame:
    """Coerce transformer input columns to numeric at fit and transform time."""
    frame = pd.DataFrame(dataframe).copy()
    for column_name in frame.columns:
        frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
    return frame


def _coerce_categorical_frame_for_sklearn(dataframe: Any) -> pd.DataFrame:
    """Coerce transformer input columns to object with explicit np.nan missingness."""
    frame = pd.DataFrame(dataframe).copy()
    for column_name in frame.columns:
        series = frame[column_name].astype(object)
        frame[column_name] = series.mask(pd.isna(series), np.nan)
    return frame


def _is_all_missing(series: pd.Series) -> bool:
    """Return True when a train feature column is entirely missing."""
    return bool(series.isna().all())


def _is_constant_after_coercion(series: pd.Series) -> bool:
    """Return True when a train feature column has only one observed value."""
    observed = series.dropna()
    if observed.empty:
        return True
    return bool(observed.nunique(dropna=True) <= 1)


def _validate_feature_list_contract(
    *,
    feature_list_payload: dict[str, Any],
    feature_columns: list[str],
    benchmark_columns: list[str],
) -> None:
    """Ensure the persisted feature-list JSON matches the dataset columns."""
    feature_list_features = sorted(feature_list_payload.get("feature_columns", []))
    feature_list_benchmarks = sorted(
        feature_list_payload.get("benchmark_only_columns", [])
    )
    if sorted(feature_columns) != feature_list_features:
        raise DataLoadError(
            "Feature-list artifact does not match the dataset feature__ columns."
        )
    if sorted(benchmark_columns) != feature_list_benchmarks:
        raise DataLoadError(
            "Feature-list artifact does not match the dataset benchmark__ columns."
        )


def _assert_split_alignment(
    *,
    encounter_medication_dataset: pd.DataFrame,
    splits: pd.DataFrame,
) -> None:
    """Ensure the dataset and split artifacts align one-to-one on the row key."""
    merged = encounter_medication_dataset.merge(
        splits.loc[:, DATASET_KEY_COLUMNS],
        how="outer",
        on=DATASET_KEY_COLUMNS,
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise DataLoadError(
            "Phase 3 baseline requires one-to-one alignment between dataset and split artifacts."
        )


def _assert_no_infinite_feature_values(
    *,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Reject infinite values unless an explicit handling path exists."""
    offending_columns: list[str] = []
    for column_name in feature_columns:
        series = dataframe[column_name]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric_series = pd.to_numeric(series, errors="coerce")
        else:
            numeric_series = pd.to_numeric(series, errors="coerce")
        array = numeric_series.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(array).any():
            offending_columns.append(column_name)
    if offending_columns:
        raise DataLoadError(
            "Phase 3 baseline does not have an infinite-value handling path for: "
            + ", ".join(sorted(offending_columns))
        )


def _build_row_id(row: pd.Series) -> str:
    """Build a stable audit row identifier from the analytical grain."""
    parts: list[str] = []
    for column_name in DATASET_KEY_COLUMNS:
        value = row[column_name]
        if isinstance(value, pd.Timestamp):
            value_text = value.isoformat()
        else:
            value_text = str(value)
        parts.append(f"{column_name}={value_text}")
    return "|".join(parts)


def _stable_run_id(payload: dict[str, Any]) -> str:
    """Create a deterministic run identifier from stable inputs."""
    digest = hashlib.sha256(
        json.dumps(_json_ready(payload), sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"first-scope-baseline-{digest}"


def _single_text_value(series: pd.Series) -> str:
    """Return the single stable text value expected in a lineage column."""
    unique_values = sorted({str(value) for value in series.dropna().astype(str).tolist()})
    if len(unique_values) != 1:
        raise DataLoadError("Expected exactly one stable lineage value.")
    return unique_values[0]


def _json_ready(value: Any) -> Any:
    """Recursively convert pandas/numpy values into JSON-safe builtins."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
