"""Minimal ordinal supervised experiment on top of clinician-approved sidecar targets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from opti_med.data_access.exceptions import DataLoadError
from opti_med.evaluation.contracts import SplitPartition
from opti_med.modeling.first_scope_baseline import (
    ROW_ID_COLUMN,
    SUSPICIOUS_FEATURE_TOKENS,
    _assert_no_infinite_feature_values,
    _fit_preprocessing_bundle,
    _single_text_value,
)
from opti_med.modeling.first_scope_dataset import (
    DATASET_KEY_COLUMNS,
    select_trainable_feature_columns_from_dataset,
    validate_first_scope_dataset_artifact,
)
from opti_med.modeling.first_scope_ordinal_targets import (
    FIRST_SCOPE_ORDINAL_TARGET_ARTIFACT_VERSION,
    ORDINAL_TARGET_DEFAULT_INCLUDE_COLUMN,
    ORDINAL_TARGET_LEVEL_COLUMN,
    ORDINAL_TARGET_LEVEL_INDEX_COLUMN,
    attach_first_scope_ordinal_targets,
)
from opti_med.scoring.priority_levels import PRIORITY_LEVELS, build_priority_level_contract_payload
from opti_med.scoring.priority_levels import derive_priority_level_series


FIRST_SCOPE_ORDINAL_BASELINE_ARTIFACT_VERSION = (
    "first_scope_ordinal_baseline_v1_65plus_first_scope.v1"
)
ORDINAL_TARGET_CONTRACT_VERSION_COLUMN = "target__artifact_contract_version"
TARGET_SCORE_COLUMN = "target__ordinal_priority_score"
SPLIT_COLUMN = "split_partition"
AUDIT_COLUMNS = [
    ROW_ID_COLUMN,
    *DATASET_KEY_COLUMNS,
    "meta__hadm_id",
    "meta__stay_id",
    SPLIT_COLUMN,
    ORDINAL_TARGET_LEVEL_COLUMN,
    ORDINAL_TARGET_LEVEL_INDEX_COLUMN,
    ORDINAL_TARGET_DEFAULT_INCLUDE_COLUMN,
]
LEVEL_INDEX_TO_LABEL = {0: "low", 1: "medium", 2: "high"}


@dataclass(frozen=True, slots=True)
class FirstScopeOrdinalSupervisedData:
    """Joined ordinal-target rows plus trainable features."""

    rows: pd.DataFrame
    feature_columns: list[str]
    dataset_build_run_id: str
    split_build_run_id: str
    ordinal_target_contract_version: str

    @property
    def y(self) -> pd.Series:
        """Return the ordinal class index target."""
        return (
            pd.to_numeric(self.rows[ORDINAL_TARGET_LEVEL_INDEX_COLUMN], errors="coerce")
            .astype("Int64")
            .astype(int)
        )


@dataclass(frozen=True, slots=True)
class FirstScopeOrdinalBaselineRunResult:
    """Outputs from one minimal ordinal baseline run."""

    run_id: str
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    feature_manifest: dict[str, Any]
    report_markdown: str


def build_first_scope_ordinal_scored_universe(
    *,
    encounter_medication_dataset: pd.DataFrame,
    ordinal_targets: pd.DataFrame,
    random_seed: int,
) -> pd.DataFrame:
    """Fit the real ordinal logistic model and score the full first-scope universe."""
    supervised_data = build_first_scope_ordinal_supervised_data(
        encounter_medication_dataset=encounter_medication_dataset,
        ordinal_targets=ordinal_targets,
    )
    train_rows = supervised_data.rows.loc[
        supervised_data.rows[SPLIT_COLUMN] == SplitPartition.TRAIN.value
    ].copy()
    train_y = (
        pd.to_numeric(train_rows[ORDINAL_TARGET_LEVEL_INDEX_COLUMN], errors="coerce")
        .astype("Int64")
        .astype(int)
    )
    preprocessor_bundle = _fit_preprocessing_bundle(
        train_frame=train_rows.loc[:, supervised_data.feature_columns],
        feature_columns=supervised_data.feature_columns,
    )
    skip_reason = _validate_ordinal_estimator_viability(
        estimator_name="logistic_regression_multinomial",
        train_y=train_y,
        retained_feature_columns=preprocessor_bundle["retained_feature_columns"],
    )
    if skip_reason is not None:
        raise DataLoadError(
            "Targeted blind evaluation requires a fit ordinal logistic model, but the current "
            f"training subset is not viable: {skip_reason}."
        )

    run_id = _stable_run_id(
        {
            "artifact_version": FIRST_SCOPE_ORDINAL_BASELINE_ARTIFACT_VERSION,
            "dataset_build_run_id": supervised_data.dataset_build_run_id,
            "split_build_run_id": supervised_data.split_build_run_id,
            "ordinal_target_contract_version": supervised_data.ordinal_target_contract_version,
            "random_seed": random_seed,
            "feature_columns": supervised_data.feature_columns,
            "scored_universe": True,
        }
    )
    estimator_spec = next(
        spec
        for spec in _build_ordinal_estimator_specs(random_seed=random_seed)
        if spec["name"] == "logistic_regression_multinomial"
    )
    estimator = estimator_spec["factory"](preprocessor_bundle["preprocessor"])
    estimator.fit(train_rows.loc[:, supervised_data.feature_columns], train_y.to_numpy(dtype=int))

    full_rows = attach_first_scope_ordinal_targets(
        encounter_medication_dataset=encounter_medication_dataset,
        ordinal_targets=ordinal_targets,
    )
    split_frame = ordinal_targets.loc[
        :,
        [
            "modeling__row_id",
            "split_partition",
            "split_build_run_id",
            "label__clinician_review_status",
            "review_submission_id",
            "review_version",
        ],
    ].drop_duplicates(subset=["modeling__row_id"])
    full_rows = full_rows.merge(
        split_frame,
        how="left",
        on="modeling__row_id",
        validate="one_to_one",
        suffixes=("", "_target"),
    )
    if "benchmark__current_rule_score" in full_rows.columns and (
        "benchmark__current_rule_score_level" not in full_rows.columns
    ):
        full_rows["benchmark__current_rule_score_level"] = derive_priority_level_series(
            full_rows["benchmark__current_rule_score"]
        )
    probabilities = _predict_multiclass_probabilities(
        estimator=estimator,
        features=full_rows.loc[:, supervised_data.feature_columns],
    )
    predicted_label_index = estimator.predict(
        full_rows.loc[:, supervised_data.feature_columns]
    ).astype(int)
    output = full_rows.loc[
        :,
        [
            "modeling__row_id",
            "subject_id",
            "encounter_id",
            "medication_standardized",
            "review_timestamp",
            "meta__hadm_id",
            "meta__stay_id",
            "benchmark__current_rule_score",
            "benchmark__current_rule_score_level",
            ORDINAL_TARGET_LEVEL_COLUMN,
            ORDINAL_TARGET_LEVEL_INDEX_COLUMN,
            ORDINAL_TARGET_DEFAULT_INCLUDE_COLUMN,
            "label__clinician_review_status",
            "split_partition",
            TARGET_SCORE_COLUMN,
        ],
    ].copy()
    output["artifact_version"] = FIRST_SCOPE_ORDINAL_BASELINE_ARTIFACT_VERSION
    output["model_run_id"] = run_id
    output["estimator_name"] = "logistic_regression_multinomial"
    output["estimator_family"] = "linear_multiclass"
    output["prediction__ordinal_priority_level_index"] = predicted_label_index.astype(int)
    output["prediction__ordinal_priority_level"] = [
        LEVEL_INDEX_TO_LABEL[int(value)] for value in predicted_label_index.tolist()
    ]
    output["prediction__probability_low"] = probabilities[:, 0].astype(float)
    output["prediction__probability_medium"] = probabilities[:, 1].astype(float)
    output["prediction__probability_high"] = probabilities[:, 2].astype(float)
    output["prediction__medium_high_margin"] = np.abs(
        output["prediction__probability_medium"] - output["prediction__probability_high"]
    ).astype(float)
    ordered = [
        "artifact_version",
        "model_run_id",
        "estimator_name",
        "estimator_family",
        "modeling__row_id",
        "subject_id",
        "encounter_id",
        "medication_standardized",
        "review_timestamp",
        "meta__hadm_id",
        "meta__stay_id",
        "split_partition",
        ORDINAL_TARGET_LEVEL_COLUMN,
        ORDINAL_TARGET_LEVEL_INDEX_COLUMN,
        ORDINAL_TARGET_DEFAULT_INCLUDE_COLUMN,
        "label__clinician_review_status",
        TARGET_SCORE_COLUMN,
        "benchmark__current_rule_score",
        "benchmark__current_rule_score_level",
        "prediction__ordinal_priority_level_index",
        "prediction__ordinal_priority_level",
        "prediction__probability_low",
        "prediction__probability_medium",
        "prediction__probability_high",
        "prediction__medium_high_margin",
    ]
    return output.loc[:, ordered].sort_values(
        ["subject_id", "encounter_id", "medication_standardized", "review_timestamp"],
        na_position="last",
    ).reset_index(drop=True)


def build_first_scope_ordinal_supervised_data(
    *,
    encounter_medication_dataset: pd.DataFrame,
    ordinal_targets: pd.DataFrame,
) -> FirstScopeOrdinalSupervisedData:
    """Join Dataset v1 to prepared ordinal targets and keep the default trainable subset only."""
    validate_first_scope_dataset_artifact(encounter_medication_dataset)
    _validate_ordinal_target_artifact(ordinal_targets)

    feature_columns = select_trainable_feature_columns_from_dataset(
        encounter_medication_dataset
    )
    if not feature_columns:
        raise DataLoadError("Ordinal baseline requires at least one feature__ column.")

    suspicious_feature_columns = sorted(
        column_name
        for column_name in feature_columns
        if any(token in column_name for token in SUSPICIOUS_FEATURE_TOKENS)
    )
    if suspicious_feature_columns:
        raise DataLoadError(
            "Ordinal baseline refused suspicious feature__ columns that look label- or benchmark-derived: "
            + ", ".join(suspicious_feature_columns)
        )

    joined = attach_first_scope_ordinal_targets(
        encounter_medication_dataset=encounter_medication_dataset,
        ordinal_targets=ordinal_targets,
    )
    joined = joined.merge(
        ordinal_targets.loc[
            :,
            [
                "modeling__row_id",
                "split_partition",
                "split_build_run_id",
                ORDINAL_TARGET_CONTRACT_VERSION_COLUMN,
            ],
        ],
        how="left",
        on="modeling__row_id",
        validate="one_to_one",
    )
    included_mask = (
        pd.to_numeric(joined[ORDINAL_TARGET_DEFAULT_INCLUDE_COLUMN], errors="coerce")
        .fillna(0)
        .astype(int)
        .eq(1)
    )
    supervised = joined.loc[included_mask].copy()
    if supervised.empty:
        raise DataLoadError(
            "Ordinal baseline requires at least one row with target__default_training_inclusion_flag == 1."
        )
    if supervised[SPLIT_COLUMN].isna().any():
        raise DataLoadError(
            "Ordinal baseline requires a split assignment for every training-included ordinal target row."
        )
    if supervised[ORDINAL_TARGET_LEVEL_INDEX_COLUMN].isna().any():
        raise DataLoadError("Ordinal baseline requires a populated ordinal target class index.")
    if supervised["modeling__row_id"].duplicated(keep=False).any():
        raise DataLoadError("Ordinal baseline requires unique modeling__row_id values.")

    _assert_no_infinite_feature_values(
        dataframe=supervised,
        feature_columns=feature_columns,
    )
    columns = [
        *AUDIT_COLUMNS,
        TARGET_SCORE_COLUMN,
        "split_build_run_id",
        ORDINAL_TARGET_CONTRACT_VERSION_COLUMN,
        *feature_columns,
    ]
    supervised = supervised.loc[:, columns].sort_values(
        DATASET_KEY_COLUMNS,
        na_position="last",
    ).reset_index(drop=True)
    return FirstScopeOrdinalSupervisedData(
        rows=supervised,
        feature_columns=feature_columns,
        dataset_build_run_id=_single_text_value(
            encounter_medication_dataset["meta__dataset_build_run_id"]
        ),
        split_build_run_id=_single_text_value(supervised["split_build_run_id"]),
        ordinal_target_contract_version=_single_text_value(
            supervised[ORDINAL_TARGET_CONTRACT_VERSION_COLUMN]
        ),
    )


def run_first_scope_ordinal_baseline(
    *,
    supervised_data: FirstScopeOrdinalSupervisedData,
    dataset_path: Path,
    ordinal_targets_path: Path,
    random_seed: int,
) -> FirstScopeOrdinalBaselineRunResult:
    """Fit a tiny multiclass baseline family and render honest ordinal metrics."""
    run_id = _stable_run_id(
        {
            "artifact_version": FIRST_SCOPE_ORDINAL_BASELINE_ARTIFACT_VERSION,
            "dataset_build_run_id": supervised_data.dataset_build_run_id,
            "split_build_run_id": supervised_data.split_build_run_id,
            "ordinal_target_contract_version": supervised_data.ordinal_target_contract_version,
            "random_seed": random_seed,
            "feature_columns": supervised_data.feature_columns,
        }
    )

    split_frames = {
        split_name: supervised_data.rows.loc[
            supervised_data.rows[SPLIT_COLUMN] == split_name
        ].copy()
        for split_name in [
            SplitPartition.TRAIN.value,
            SplitPartition.VALIDATION.value,
            SplitPartition.TEST.value,
        ]
    }
    train_rows = split_frames[SplitPartition.TRAIN.value]
    train_y = (
        pd.to_numeric(train_rows[ORDINAL_TARGET_LEVEL_INDEX_COLUMN], errors="coerce")
        .astype("Int64")
        .astype(int)
    )

    preprocessor_bundle = _fit_preprocessing_bundle(
        train_frame=train_rows.loc[:, supervised_data.feature_columns],
        feature_columns=supervised_data.feature_columns,
    )

    estimator_results: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []
    for estimator_spec in _build_ordinal_estimator_specs(random_seed=random_seed):
        estimator_payload = _fit_and_evaluate_ordinal_estimator(
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

    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else _empty_prediction_artifact_frame()
    )
    if not predictions.empty:
        predictions = predictions.sort_values(
            ["estimator_name", SPLIT_COLUMN, ROW_ID_COLUMN],
            na_position="last",
        ).reset_index(drop=True)

    feature_manifest = {
        "artifact_version": FIRST_SCOPE_ORDINAL_BASELINE_ARTIFACT_VERSION,
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
    metrics = _build_ordinal_run_metrics(
        supervised_data=supervised_data,
        run_id=run_id,
        dataset_path=dataset_path,
        ordinal_targets_path=ordinal_targets_path,
        random_seed=random_seed,
        feature_manifest=feature_manifest,
        estimator_results=estimator_results,
    )
    report_markdown = build_first_scope_ordinal_baseline_report(metrics=metrics)
    return FirstScopeOrdinalBaselineRunResult(
        run_id=run_id,
        predictions=predictions,
        metrics=metrics,
        feature_manifest=feature_manifest,
        report_markdown=report_markdown,
    )


def write_first_scope_ordinal_baseline_artifacts(
    *,
    result: FirstScopeOrdinalBaselineRunResult,
    output_root: Path,
    report_output: Path | None = None,
) -> dict[str, Path]:
    """Persist the minimal ordinal baseline artifacts."""
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": output_root / "predictions.parquet",
        "metrics": output_root / "metrics.json",
        "feature_manifest": output_root / "feature_manifest.json",
        "report": output_root / "modeling_report.md",
    }
    result.predictions.to_parquet(paths["predictions"], index=False)
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


def build_first_scope_ordinal_baseline_report(*, metrics: dict[str, Any]) -> str:
    """Render a compact QC report for the minimal ordinal supervised experiment."""
    lines = [
        "# Phase 8 Minimal Ordinal Supervised Experiment Report",
        "",
        "This is a feasibility run on the clinician-approved ordinal sidecar. It is not a model-readiness or performance milestone.",
        "",
        "## Input Artifacts",
        "",
        f"- dataset: `{metrics['input_artifacts']['dataset_path']}`",
        f"- ordinal_targets: `{metrics['input_artifacts']['ordinal_targets_path']}`",
        f"- dataset_build_run_id: `{metrics['dataset_build_run_id']}`",
        f"- split_build_run_id: `{metrics['split_build_run_id']}`",
        f"- ordinal_baseline_run_id: `{metrics['run_id']}`",
        "",
        "## Training Contract",
        "",
        f"- row filter: `{metrics['training_contract']['row_filter_expression']}`",
        f"- supervised target: `{metrics['training_contract']['label_column']}`",
        f"- class order: `{', '.join(metrics['training_contract']['ordered_levels'])}`",
        f"- trainable inputs: `{metrics['training_contract']['trainable_feature_prefix']}*` only",
        "",
        "## Supervised Subset",
        "",
        f"- eligible rows used for modeling: {metrics['eligible_row_count']:,}",
        f"- retained raw feature count: {metrics['retained_feature_count']:,}",
        f"- transformed feature count: {metrics['transformed_feature_count']:,}",
        f"- benchmark current-rule populated rows on modeling subset: {metrics['benchmark_current_rule_populated_row_count']:,}",
        "",
        "### Split Counts",
        "",
    ]
    for split_name, payload in metrics["split_summary"].items():
        level_counts = ", ".join(
            f"{level}={payload['level_counts'].get(level, 0):,}" for level in PRIORITY_LEVELS
        )
        lines.append(
            f"- {split_name}: rows={payload['row_count']:,}, {level_counts}"
        )
    lines.extend(
        [
            "",
            "## Estimator Status",
            "",
        ]
    )
    for estimator_name, payload in metrics["estimators"].items():
        reason = payload["skip_reason"] or "fit"
        lines.append(f"- {estimator_name}: {payload['status']} ({reason})")
    lines.extend(
        [
            "",
            "## Estimator Metrics",
            "",
            "| estimator | split | rows | accuracy | balanced_acc | macro_precision | macro_recall | macro_f1 | weighted_kappa | mean_abs_level_error | off_by_one | severe | notes |",
            "| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
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
            lines.append(
                "| "
                + " | ".join(
                    [
                        estimator_name,
                        split_name,
                        f"{split_payload['row_count']:,}",
                        _format_optional_metric(split_payload["metrics"]["accuracy"]),
                        _format_optional_metric(split_payload["metrics"]["balanced_accuracy"]),
                        _format_optional_metric(split_payload["metrics"]["macro_precision"]),
                        _format_optional_metric(split_payload["metrics"]["macro_recall"]),
                        _format_optional_metric(split_payload["metrics"]["macro_f1"]),
                        _format_optional_metric(split_payload["metrics"]["weighted_kappa_quadratic"]),
                        _format_optional_metric(split_payload["metrics"]["mean_absolute_level_error"]),
                        _format_optional_metric(split_payload["metrics"]["off_by_one_rate"]),
                        _format_optional_metric(split_payload["metrics"]["severe_disagreement_rate"]),
                        _format_split_notes(split_payload["warnings"]),
                    ]
                )
                + " |"
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


def _validate_ordinal_target_artifact(ordinal_targets: pd.DataFrame) -> None:
    required_columns = [
        "modeling__row_id",
        ORDINAL_TARGET_LEVEL_COLUMN,
        ORDINAL_TARGET_LEVEL_INDEX_COLUMN,
        ORDINAL_TARGET_DEFAULT_INCLUDE_COLUMN,
        "split_partition",
        "split_build_run_id",
        ORDINAL_TARGET_CONTRACT_VERSION_COLUMN,
    ]
    missing_columns = sorted(set(required_columns) - set(ordinal_targets.columns))
    if missing_columns:
        raise DataLoadError(
            "Ordinal target sidecar is missing required columns: "
            + ", ".join(missing_columns)
        )


def _build_ordinal_estimator_specs(*, random_seed: int) -> list[dict[str, Any]]:
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
            "name": "logistic_regression_multinomial",
            "family": "linear_multiclass",
            "factory": lambda preprocessor: Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=0.1,
                            solver="lbfgs",
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=random_seed,
                        ),
                    ),
                ]
            ),
        },
    ]


def _fit_and_evaluate_ordinal_estimator(
    *,
    estimator_spec: dict[str, Any],
    supervised_data: FirstScopeOrdinalSupervisedData,
    train_y: pd.Series,
    split_frames: dict[str, pd.DataFrame],
    preprocessor_bundle: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    skip_reason = _validate_ordinal_estimator_viability(
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
    estimator.fit(train_X, train_y.to_numpy(dtype=int))

    split_metrics: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []
    for split_name, split_rows in split_frames.items():
        if split_rows.empty:
            split_metrics[split_name] = _empty_ordinal_split_payload("no_rows_available")
            continue
        y_true = (
            pd.to_numeric(split_rows[ORDINAL_TARGET_LEVEL_INDEX_COLUMN], errors="coerce")
            .astype("Int64")
            .astype(int)
            .to_numpy(dtype=int)
        )
        split_X = (
            _dummy_feature_frame(split_rows)
            if estimator_spec["family"] == "dummy"
            else split_rows.loc[:, supervised_data.feature_columns]
        )
        y_pred = estimator.predict(split_X).astype(int)
        probabilities = _predict_multiclass_probabilities(estimator=estimator, features=split_X)
        split_metrics[split_name] = _evaluate_ordinal_predictions(
            y_true=y_true,
            y_pred=np.asarray(y_pred, dtype=int),
        )
        prediction_frames.append(
            _build_ordinal_prediction_frame(
                split_rows=split_rows,
                estimator_name=estimator_spec["name"],
                estimator_family=estimator_spec["family"],
                run_id=run_id,
                predicted_label_index=np.asarray(y_pred, dtype=int),
                predicted_probabilities=probabilities,
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


def _validate_ordinal_estimator_viability(
    *,
    estimator_name: str,
    train_y: pd.Series,
    retained_feature_columns: list[str],
) -> str | None:
    if train_y.empty:
        return "training split has no ordinal supervision rows"
    class_counts = train_y.value_counts().sort_index()
    unique_class_count = int(class_counts.index.nunique())
    if estimator_name == "dummy_stratified" and unique_class_count < 2:
        return "stratified dummy requires at least two classes in the training split"
    if estimator_name == "logistic_regression_multinomial":
        if unique_class_count < 2:
            return "multinomial logistic regression requires at least two classes in the training split"
        if int(class_counts.min()) < 2:
            return (
                "multinomial logistic regression requires at least two training rows in the rarest observed class for a minimally stable pilot"
            )
        if not retained_feature_columns:
            return "all trainable features were dropped during train-only preprocessing"
    return None


def _build_ordinal_prediction_frame(
    *,
    split_rows: pd.DataFrame,
    estimator_name: str,
    estimator_family: str,
    run_id: str,
    predicted_label_index: np.ndarray,
    predicted_probabilities: np.ndarray,
) -> pd.DataFrame:
    output = split_rows.loc[:, AUDIT_COLUMNS + [TARGET_SCORE_COLUMN]].copy()
    output["artifact_version"] = FIRST_SCOPE_ORDINAL_BASELINE_ARTIFACT_VERSION
    output["model_run_id"] = run_id
    output["estimator_name"] = estimator_name
    output["estimator_family"] = estimator_family
    output["prediction__ordinal_priority_level_index"] = predicted_label_index.astype(int)
    output["prediction__ordinal_priority_level"] = [
        LEVEL_INDEX_TO_LABEL[int(value)] for value in predicted_label_index.tolist()
    ]
    for index, level in LEVEL_INDEX_TO_LABEL.items():
        output[f"prediction__probability_{level}"] = predicted_probabilities[:, index].astype(float)
    ordered = [
        "artifact_version",
        "model_run_id",
        "estimator_name",
        "estimator_family",
        *AUDIT_COLUMNS,
        TARGET_SCORE_COLUMN,
        "prediction__ordinal_priority_level_index",
        "prediction__ordinal_priority_level",
        "prediction__probability_low",
        "prediction__probability_medium",
        "prediction__probability_high",
    ]
    return output.loc[:, ordered]


def _predict_multiclass_probabilities(*, estimator: Any, features: pd.DataFrame) -> np.ndarray:
    if not hasattr(estimator, "predict_proba"):
        return np.full((len(features), len(PRIORITY_LEVELS)), np.nan, dtype=float)
    probabilities = estimator.predict_proba(features)
    classes = getattr(estimator, "classes_", None)
    if classes is None and hasattr(estimator, "named_steps"):
        classes = getattr(estimator.named_steps.get("model"), "classes_", None)
    if classes is None:
        raise ValueError("Estimator does not expose classes_ for multiclass probability alignment.")
    output = np.zeros((len(features), len(PRIORITY_LEVELS)), dtype=float)
    for class_index, class_value in enumerate(np.asarray(classes, dtype=int).tolist()):
        output[:, int(class_value)] = probabilities[:, class_index]
    return output


def _evaluate_ordinal_predictions(*, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    row_count = int(len(y_true))
    if row_count == 0:
        return _empty_ordinal_split_payload("no_rows_available")

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    warnings: list[str] = []
    accuracy = float(np.mean(y_true == y_pred))

    if len(np.unique(y_true)) >= 2:
        balanced_accuracy = _balanced_accuracy_present_classes(y_true=y_true, y_pred=y_pred)
        weighted_kappa = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    else:
        balanced_accuracy = None
        weighted_kappa = None
        warnings.append("Ground truth contains only one class; balanced accuracy and weighted kappa are not informative.")

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
    absolute_errors = np.abs(y_pred - y_true)
    mean_absolute_level_error = float(np.mean(absolute_errors))
    off_by_one_rate = float(np.mean(absolute_errors == 1))
    severe_disagreement_rate = float(np.mean(absolute_errors >= 2))
    if row_count < 10:
        warnings.append("Split size is very small; held-out ordinal metrics are high-variance.")
    return {
        "row_count": row_count,
        "level_counts": _level_count_dict(pd.Series(y_true)),
        "metrics": {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_kappa_quadratic": weighted_kappa,
            "mean_absolute_level_error": mean_absolute_level_error,
            "off_by_one_rate": off_by_one_rate,
            "severe_disagreement_rate": severe_disagreement_rate,
        },
        "warnings": sorted(set(warnings)),
    }


def _empty_ordinal_split_payload(reason: str) -> dict[str, Any]:
    return {
        "row_count": 0,
        "level_counts": _level_count_dict(pd.Series(dtype="Int64")),
        "metrics": {
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "weighted_kappa_quadratic": None,
            "mean_absolute_level_error": None,
            "off_by_one_rate": None,
            "severe_disagreement_rate": None,
        },
        "warnings": [reason],
    }


def _empty_split_metrics() -> dict[str, Any]:
    return {
        split_name: _empty_ordinal_split_payload("not_run")
        for split_name in [
            SplitPartition.TRAIN.value,
            SplitPartition.VALIDATION.value,
            SplitPartition.TEST.value,
        ]
    }


def _build_ordinal_run_metrics(
    *,
    supervised_data: FirstScopeOrdinalSupervisedData,
    run_id: str,
    dataset_path: Path,
    ordinal_targets_path: Path,
    random_seed: int,
    feature_manifest: dict[str, Any],
    estimator_results: dict[str, Any],
) -> dict[str, Any]:
    split_summary: dict[str, Any] = {}
    warnings: set[str] = set()
    for split_name in [
        SplitPartition.TRAIN.value,
        SplitPartition.VALIDATION.value,
        SplitPartition.TEST.value,
    ]:
        split_rows = supervised_data.rows.loc[
            supervised_data.rows[SPLIT_COLUMN] == split_name
        ].copy()
        level_series = pd.to_numeric(
            split_rows[ORDINAL_TARGET_LEVEL_INDEX_COLUMN],
            errors="coerce",
        ).astype("Int64")
        split_summary[split_name] = {
            "row_count": int(len(split_rows)),
            "level_counts": _level_count_dict(level_series),
        }
        if len(split_rows) < 5:
            warnings.add(
                f"{split_name} split has fewer than five ordinal target rows; held-out estimates are extremely unstable."
            )
    if split_summary[SplitPartition.TEST.value]["row_count"] < 5:
        warnings.add(
            "This ordinal pilot has only two held-out test rows, so any observed test performance is directional only."
        )
    warnings.add(
        "No populated benchmark__current_rule_score rows are available on the ordinal modeling subset, so there is no rule baseline comparison in this experiment."
    )
    warnings.add(
        "This run uses clinician ordinal levels as the canonical supervised target and treats optional clinician numeric scores as diagnostic only."
    )
    return {
        "artifact_version": FIRST_SCOPE_ORDINAL_BASELINE_ARTIFACT_VERSION,
        "run_id": run_id,
        "random_seed": random_seed,
        "input_artifacts": {
            "dataset_path": str(dataset_path),
            "ordinal_targets_path": str(ordinal_targets_path),
        },
        "training_contract": {
            "row_filter_expression": "target__default_training_inclusion_flag == 1",
            "label_column": ORDINAL_TARGET_LEVEL_COLUMN,
            "label_index_column": ORDINAL_TARGET_LEVEL_INDEX_COLUMN,
            "trainable_feature_prefix": "feature__",
            "ordered_levels": list(PRIORITY_LEVELS),
        },
        "dataset_build_run_id": supervised_data.dataset_build_run_id,
        "split_build_run_id": supervised_data.split_build_run_id,
        "ordinal_target_contract_version": supervised_data.ordinal_target_contract_version,
        "eligible_row_count": int(len(supervised_data.rows)),
        "retained_feature_count": feature_manifest["retained_raw_feature_count"],
        "transformed_feature_count": feature_manifest["transformed_feature_count"],
        "benchmark_current_rule_populated_row_count": int(
            supervised_data.rows["benchmark__current_rule_score"].notna().sum()
        )
        if "benchmark__current_rule_score" in supervised_data.rows.columns
        else 0,
        "split_summary": split_summary,
        "estimators": estimator_results,
        "priority_level_contract": build_priority_level_contract_payload(),
        "feature_manifest": feature_manifest,
        "warnings": sorted(warnings),
    }


def _level_count_dict(level_index_series: pd.Series) -> dict[str, int]:
    normalized = pd.to_numeric(level_index_series, errors="coerce").astype("Int64")
    return {
        level: int(normalized.eq(index).sum())
        for index, level in LEVEL_INDEX_TO_LABEL.items()
    }


def _dummy_feature_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"__dummy__": np.zeros(len(dataframe), dtype=float)})


def _empty_prediction_artifact_frame() -> pd.DataFrame:
    ordered = [
        "artifact_version",
        "model_run_id",
        "estimator_name",
        "estimator_family",
        *AUDIT_COLUMNS,
        TARGET_SCORE_COLUMN,
        "prediction__ordinal_priority_level_index",
        "prediction__ordinal_priority_level",
        "prediction__probability_low",
        "prediction__probability_medium",
        "prediction__probability_high",
    ]
    return pd.DataFrame(columns=ordered)


def _format_optional_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _format_split_notes(warnings: list[str]) -> str:
    if not warnings:
        return "none"
    return "; ".join(sorted(set(warnings)))


def _balanced_accuracy_present_classes(*, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute balanced accuracy over the classes that are present in y_true only."""
    recalls: list[float] = []
    for class_value in sorted(set(np.asarray(y_true, dtype=int).tolist())):
        class_mask = y_true == class_value
        denominator = int(class_mask.sum())
        if denominator == 0:
            continue
        recalls.append(float((y_pred[class_mask] == class_value).sum()) / float(denominator))
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def _stable_run_id(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(_json_ready(payload), sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"first-scope-ordinal-baseline-{digest}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value
