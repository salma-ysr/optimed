"""Ordinal target preparation from latest-active clinician reviews."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from opti_med.data_access.exceptions import DataLoadError
from opti_med.evaluation.contracts import SplitPartition
from opti_med.modeling.first_scope_dataset import (
    DATASET_KEY_COLUMNS,
    validate_first_scope_dataset_artifact,
    validate_first_scope_split_artifact,
)
from opti_med.scoring.priority_levels import (
    PRIORITY_LEVEL_ORDER,
    PRIORITY_LEVELS,
    PRIORITY_SCORE_SEMANTIC_FAMILY,
    coerce_priority_score_value,
    derive_priority_level_series,
    normalize_priority_level,
    resolve_priority_level_series,
)


FIRST_SCOPE_ORDINAL_TARGET_ARTIFACT_VERSION = (
    "encounter_medication_ordinal_targets_v1_65plus_first_scope.v1"
)
ORDINAL_TARGET_SOURCE_NAME = "latest_active_clinician_review_phase6"
ORDINAL_TARGET_LEVEL_COLUMN = "target__ordinal_priority_level"
ORDINAL_TARGET_LEVEL_INDEX_COLUMN = "target__ordinal_priority_level_index"
ORDINAL_TARGET_DEFAULT_INCLUDE_COLUMN = "target__default_training_inclusion_flag"
ORDINAL_TARGET_DEFAULT_EXCLUSION_REASON_COLUMN = (
    "target__default_training_exclusion_reason"
)
ORDINAL_TARGET_COLUMNS = [
    "modeling__row_id",
    *DATASET_KEY_COLUMNS,
    "meta__hadm_id",
    "meta__stay_id",
    "benchmark__medication_class_only_medication_class_standardized",
    "benchmark__current_rule_score",
    "benchmark__current_rule_score_level",
    "split_partition",
    "split_build_run_id",
    "meta__feature_available_as_of_review_time_flag",
    "meta__dataset_build_run_id",
    "meta__dataset_contract_version",
    "review_submission_id",
    "review_submission_timestamp",
    "review_version",
    "reviewer_id",
    "label__clinician_priority_level",
    "label__clinician_priority_score",
    "label__clinician_priority_score_level",
    "label__clinician_review_status",
    "label__clinician_reason_tags",
    "label__clinician_note",
    "label__clinician_suggested_action",
    ORDINAL_TARGET_LEVEL_COLUMN,
    ORDINAL_TARGET_LEVEL_INDEX_COLUMN,
    "target__ordinal_priority_score",
    "target__ordinal_priority_score_level",
    "target__priority_semantic_family",
    "target__label_source_name",
    "target__level_available_flag",
    "target__numeric_score_available_flag",
    "target__score_level_alignment_flag",
    "target__phase4_level_comparison_ready_flag",
    "target__phase4_score_comparison_ready_flag",
    ORDINAL_TARGET_DEFAULT_INCLUDE_COLUMN,
    ORDINAL_TARGET_DEFAULT_EXCLUSION_REASON_COLUMN,
    "target__artifact_contract_version",
]
REQUIRED_CLINICIAN_REVIEW_COLUMNS = [
    "modeling__row_id",
    "subject_id",
    "encounter_id",
    "medication_standardized",
    "review_timestamp",
    "review_submission_id",
    "review_version",
    "label__clinician_priority_level",
    "label__clinician_review_status",
]


@dataclass(frozen=True, slots=True)
class FirstScopeOrdinalTargetBuildResult:
    """Prepared ordinal target artifact plus QC/report outputs."""

    targets: pd.DataFrame
    summary: dict[str, Any]
    report_markdown: str


def build_first_scope_ordinal_targets(
    *,
    encounter_medication_dataset: pd.DataFrame,
    latest_clinician_reviews: pd.DataFrame,
    splits: pd.DataFrame | None = None,
) -> FirstScopeOrdinalTargetBuildResult:
    """Prepare the first-scope ordinal target sidecar from latest-active clinician reviews."""
    validate_first_scope_dataset_artifact(encounter_medication_dataset)
    if splits is not None:
        validate_first_scope_split_artifact(splits)

    dataset = _prepare_dataset_frame(encounter_medication_dataset)
    latest = _prepare_latest_reviews_frame(latest_clinician_reviews)

    if latest.empty:
        empty_targets = pd.DataFrame(columns=ORDINAL_TARGET_COLUMNS)
        summary = _build_ordinal_target_summary(
            targets=empty_targets,
            total_latest_review_rows=0,
            score_level_mismatch_count=0,
        )
        return FirstScopeOrdinalTargetBuildResult(
            targets=empty_targets,
            summary=summary,
            report_markdown=build_first_scope_ordinal_target_report(summary=summary),
        )

    if latest["modeling__row_id"].duplicated(keep=False).any():
        raise DataLoadError(
            "Latest clinician review snapshot contains duplicate modeling__row_id values."
        )

    targets = latest.merge(
        dataset,
        how="left",
        on="modeling__row_id",
        validate="one_to_one",
        suffixes=("", "_dataset"),
    )
    dataset_presence_column = f"{DATASET_KEY_COLUMNS[0]}_dataset"
    if targets[dataset_presence_column].isna().any():
        unmatched_count = int(targets[dataset_presence_column].isna().sum())
        raise DataLoadError(
            "Latest clinician reviews could not all be reconciled to Dataset v1 by modeling__row_id "
            f"({unmatched_count} unmatched rows)."
        )
    _assert_key_alignment(targets)

    if splits is not None:
        split_frame = _prepare_split_frame(splits)
        targets = targets.merge(
            split_frame,
            how="left",
            on="modeling__row_id",
            validate="one_to_one",
        )
        if targets["split_partition"].isna().any():
            raise DataLoadError(
                "Every prepared ordinal target row must align to exactly one persisted split assignment."
            )
    else:
        targets["split_partition"] = pd.NA
        targets["split_build_run_id"] = pd.NA

    target_level = targets["label__clinician_priority_level"].apply(normalize_priority_level)
    score_series = targets["label__clinician_priority_score"].apply(coerce_priority_score_value)
    score_level = resolve_priority_level_series(
        score_series=score_series,
        level_series=targets["label__clinician_priority_score_level"],
        score_name="label__clinician_priority_score",
        level_name="label__clinician_priority_score_level",
    )
    target_level = target_level.fillna(score_level).astype("string")
    score_alignment_flag = score_level.isna() | target_level.eq(score_level)

    targets[ORDINAL_TARGET_LEVEL_COLUMN] = target_level
    targets[ORDINAL_TARGET_LEVEL_INDEX_COLUMN] = target_level.map(PRIORITY_LEVEL_ORDER).astype(
        "Int64"
    )
    targets["target__ordinal_priority_score"] = score_series
    targets["target__ordinal_priority_score_level"] = score_level.astype("string")
    targets["target__priority_semantic_family"] = PRIORITY_SCORE_SEMANTIC_FAMILY
    targets["target__label_source_name"] = ORDINAL_TARGET_SOURCE_NAME
    targets["target__level_available_flag"] = target_level.notna().astype(int)
    targets["target__numeric_score_available_flag"] = score_series.notna().astype(int)
    targets["target__score_level_alignment_flag"] = score_alignment_flag.astype(int)
    targets["target__phase4_level_comparison_ready_flag"] = target_level.notna().astype(int)
    targets["target__phase4_score_comparison_ready_flag"] = score_series.notna().astype(int)
    targets[ORDINAL_TARGET_DEFAULT_INCLUDE_COLUMN] = targets.apply(
        _default_training_inclusion_flag,
        axis=1,
    ).astype(int)
    targets[ORDINAL_TARGET_DEFAULT_EXCLUSION_REASON_COLUMN] = targets.apply(
        _default_training_exclusion_reason,
        axis=1,
    ).astype("string")
    targets["target__artifact_contract_version"] = FIRST_SCOPE_ORDINAL_TARGET_ARTIFACT_VERSION

    for column_name in ORDINAL_TARGET_COLUMNS:
        if column_name not in targets.columns:
            targets[column_name] = pd.NA
    targets = (
        targets.loc[:, ORDINAL_TARGET_COLUMNS]
        .sort_values(
            ["subject_id", "encounter_id", "review_timestamp", "medication_standardized"],
            na_position="last",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    summary = _build_ordinal_target_summary(
        targets=targets,
        total_latest_review_rows=int(len(latest)),
        score_level_mismatch_count=int((~score_alignment_flag).sum()),
    )
    return FirstScopeOrdinalTargetBuildResult(
        targets=targets,
        summary=summary,
        report_markdown=build_first_scope_ordinal_target_report(summary=summary),
    )


def write_first_scope_ordinal_target_artifacts(
    *,
    result: FirstScopeOrdinalTargetBuildResult,
    output_path: Path,
    summary_path: Path,
    report_path: Path,
) -> dict[str, Path]:
    """Persist the ordinal target sidecar and QC outputs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result.targets.to_parquet(output_path, index=False)
    summary_payload = dict(result.summary)
    artifact_paths = dict(summary_payload.get("artifact_paths", {}))
    artifact_paths.update(
        {
            "ordinal_targets_output_path": str(output_path),
            "ordinal_targets_summary_path": str(summary_path),
            "ordinal_targets_report_path": str(report_path),
        }
    )
    summary_payload["artifact_paths"] = artifact_paths
    summary_path.write_text(
        json.dumps(_json_ready(summary_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(result.report_markdown, encoding="utf-8")
    return {
        "targets": output_path,
        "summary": summary_path,
        "report": report_path,
    }


def build_first_scope_ordinal_target_report(*, summary: dict[str, Any]) -> str:
    """Render a compact markdown QC report for the prepared ordinal targets."""
    lines = [
        "# Phase 7 Ordinal Target Preparation Report",
        "",
        "This is a target-preparation milestone built from clinician-reviewed labels. It is not evidence of model readiness or external validation.",
        "",
        "## Input Artifacts",
        "",
        f"- dataset: `{summary['artifact_paths']['dataset_path']}`",
        f"- clinician review snapshot: `{summary['artifact_paths']['clinician_review_snapshot_path']}`",
        f"- splits: `{summary['artifact_paths']['splits_path']}`",
        "",
        "## Prepared Target Contract",
        "",
        f"- contract_version: `{summary['contract_version']}`",
        f"- semantic family: `{summary['target_contract']['priority_semantic_family']}`",
        f"- ordered levels: `{', '.join(summary['target_contract']['ordered_levels'])}`",
        f"- default inclusion rule: `{summary['target_contract']['default_training_filter_expression']}`",
        "",
        "## Label Base",
        "",
        f"- latest-active clinician review rows: {summary['latest_active_review_row_count']:,}",
        f"- prepared ordinal target rows: {summary['prepared_target_row_count']:,}",
        f"- default training-included rows: {summary['default_training_included_row_count']:,}",
        f"- default excluded rows: {summary['default_training_excluded_row_count']:,}",
        f"- latest reviews matched to dataset rows: {summary['matched_to_dataset_row_count']:,}",
        f"- score-level mismatch rows: {summary['score_level_mismatch_row_count']:,}",
        "",
        "## Ordinal Balance",
        "",
    ]
    for level in PRIORITY_LEVELS:
        all_count = int(summary["priority_level_frequencies_all"].get(level, 0))
        included_count = int(summary["priority_level_frequencies_default_included"].get(level, 0))
        lines.append(f"- {level}: all={all_count:,}, default_included={included_count:,}")
    lines.extend(
        [
            "",
            "## Review Status Mix",
            "",
        ]
    )
    for status_name, count in summary["review_status_frequencies"].items():
        lines.append(f"- {status_name}: {count:,}")
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- unique subjects: {summary['unique_subject_count']:,}",
            f"- unique encounters: {summary['unique_encounter_count']:,}",
            f"- medication classes: {summary['medication_class_frequencies']}",
            f"- numeric score coverage: {summary['numeric_score_row_count']:,} rows ({summary['numeric_score_coverage_rate']:.2%})",
            f"- benchmark current-rule coverage on prepared target rows: {summary['benchmark_available_row_count']:,} rows ({summary['benchmark_available_rate']:.2%})",
            "",
            "## Split Coverage",
            "",
        ]
    )
    for split_name, count in summary["split_partition_frequencies_all"].items():
        included_count = int(summary["split_partition_frequencies_default_included"].get(split_name, 0))
        lines.append(f"- {split_name}: all={count:,}, default_included={included_count:,}")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The prepared sidecar preserves all latest-active clinician labels, but the default training subset excludes non-`reviewed` statuses.",
            "- Optional clinician numeric scores remain sparse, so low/medium/high is the canonical training target for now.",
            "- This artifact is designed to join back to Dataset v1 and future Phase 4-style comparisons without overwriting any existing label__, benchmark__, or feature__ fields.",
        ]
    )
    return "\n".join(lines) + "\n"


def attach_first_scope_ordinal_targets(
    *,
    encounter_medication_dataset: pd.DataFrame,
    ordinal_targets: pd.DataFrame,
) -> pd.DataFrame:
    """Join prepared ordinal targets back onto Dataset v1 by modeling__row_id."""
    validate_first_scope_dataset_artifact(encounter_medication_dataset)
    dataset = encounter_medication_dataset.copy().reset_index(drop=True)
    dataset["modeling__row_id"] = dataset.apply(_build_modeling_row_id, axis=1)
    if dataset["modeling__row_id"].duplicated(keep=False).any():
        raise DataLoadError("Dataset v1 has duplicate modeling__row_id values.")
    if ordinal_targets.duplicated(subset=["modeling__row_id"], keep=False).any():
        raise DataLoadError("Ordinal target sidecar must have unique modeling__row_id values.")
    target_columns = [
        "modeling__row_id",
        ORDINAL_TARGET_LEVEL_COLUMN,
        ORDINAL_TARGET_LEVEL_INDEX_COLUMN,
        "target__ordinal_priority_score",
        "target__ordinal_priority_score_level",
        "label__clinician_review_status",
        ORDINAL_TARGET_DEFAULT_INCLUDE_COLUMN,
        ORDINAL_TARGET_DEFAULT_EXCLUSION_REASON_COLUMN,
        "target__priority_semantic_family",
    ]
    joined = dataset.merge(
        ordinal_targets.loc[:, target_columns],
        how="left",
        on="modeling__row_id",
        validate="one_to_one",
    )
    return joined


def _prepare_dataset_frame(encounter_medication_dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = encounter_medication_dataset.copy().reset_index(drop=True)
    dataset["modeling__row_id"] = dataset.apply(_build_modeling_row_id, axis=1)
    if dataset["modeling__row_id"].duplicated(keep=False).any():
        raise DataLoadError("Dataset v1 has duplicate modeling__row_id values.")
    if "benchmark__current_rule_score" in dataset.columns:
        dataset["benchmark__current_rule_score_level"] = derive_priority_level_series(
            dataset["benchmark__current_rule_score"]
        )
    else:
        dataset["benchmark__current_rule_score"] = pd.NA
        dataset["benchmark__current_rule_score_level"] = pd.Series(dtype="string")
    dataset["meta__feature_available_as_of_review_time_flag"] = pd.to_numeric(
        dataset["meta__feature_available_as_of_review_time_flag"],
        errors="coerce",
    ).fillna(0).astype(int)
    keep_columns = [
        "modeling__row_id",
        *DATASET_KEY_COLUMNS,
        "meta__hadm_id",
        "meta__stay_id",
        "benchmark__medication_class_only_medication_class_standardized",
        "benchmark__current_rule_score",
        "benchmark__current_rule_score_level",
        "meta__feature_available_as_of_review_time_flag",
        "meta__dataset_build_run_id",
        "meta__dataset_contract_version",
    ]
    return dataset.loc[:, keep_columns].copy()


def _prepare_split_frame(splits: pd.DataFrame) -> pd.DataFrame:
    split_frame = splits.copy().reset_index(drop=True)
    split_frame["modeling__row_id"] = split_frame.apply(_build_modeling_row_id, axis=1)
    if split_frame["modeling__row_id"].duplicated(keep=False).any():
        raise DataLoadError("Split artifact has duplicate modeling__row_id values.")
    return split_frame.loc[:, ["modeling__row_id", "split_partition", "split_build_run_id"]].copy()


def _prepare_latest_reviews_frame(latest_clinician_reviews: pd.DataFrame) -> pd.DataFrame:
    latest = latest_clinician_reviews.copy().reset_index(drop=True)
    missing_columns = sorted(
        set(REQUIRED_CLINICIAN_REVIEW_COLUMNS) - set(latest.columns)
    )
    if missing_columns:
        raise DataLoadError(
            "Latest clinician review snapshot is missing required columns: "
            + ", ".join(missing_columns)
        )
    latest["label__clinician_priority_level"] = latest["label__clinician_priority_level"].apply(
        normalize_priority_level
    )
    latest["label__clinician_review_status"] = (
        latest["label__clinician_review_status"].fillna("").astype(str).str.strip().str.lower()
    )
    if "label__clinician_priority_score" not in latest.columns:
        latest["label__clinician_priority_score"] = pd.NA
    latest["label__clinician_priority_score"] = latest["label__clinician_priority_score"].apply(
        coerce_priority_score_value
    )
    if "label__clinician_priority_score_level" not in latest.columns:
        latest["label__clinician_priority_score_level"] = pd.NA
    if "label__clinician_reason_tags" not in latest.columns:
        latest["label__clinician_reason_tags"] = [[] for _ in range(len(latest))]
    latest["label__clinician_reason_tags"] = latest["label__clinician_reason_tags"].apply(
        _normalize_reason_tags
    )
    for column_name in [
        "label__clinician_note",
        "label__clinician_suggested_action",
        "review_submission_timestamp",
        "reviewer_id",
    ]:
        if column_name not in latest.columns:
            latest[column_name] = pd.NA
    return latest


def _assert_key_alignment(targets: pd.DataFrame) -> None:
    for column_name in DATASET_KEY_COLUMNS:
        dataset_column = f"{column_name}_dataset"
        if dataset_column not in targets.columns:
            continue
        mismatch_mask = (
            targets[column_name].astype(str) != targets[dataset_column].astype(str)
        )
        if mismatch_mask.any():
            raise DataLoadError(
                f"Latest clinician reviews do not align to Dataset v1 for {column_name}."
            )
        targets.drop(columns=[column_name], inplace=True)
        targets.rename(columns={dataset_column: column_name}, inplace=True)


def _default_training_inclusion_flag(row: pd.Series) -> int:
    review_status = str(row.get("label__clinician_review_status") or "").strip().lower()
    feature_flag = int(pd.to_numeric(pd.Series([row.get("meta__feature_available_as_of_review_time_flag")]), errors="coerce").fillna(0).iloc[0])
    return int(
        pd.notna(row.get(ORDINAL_TARGET_LEVEL_COLUMN))
        and review_status == "reviewed"
        and feature_flag == 1
    )


def _default_training_exclusion_reason(row: pd.Series) -> str | None:
    if pd.isna(row.get(ORDINAL_TARGET_LEVEL_COLUMN)):
        return "missing_priority_level"
    feature_flag = int(pd.to_numeric(pd.Series([row.get("meta__feature_available_as_of_review_time_flag")]), errors="coerce").fillna(0).iloc[0])
    if feature_flag != 1:
        return "feature_unavailable_as_of_review_time"
    review_status = str(row.get("label__clinician_review_status") or "").strip().lower()
    if review_status != "reviewed":
        return f"review_status_{review_status or 'missing'}"
    return None


def _build_ordinal_target_summary(
    *,
    targets: pd.DataFrame,
    total_latest_review_rows: int,
    score_level_mismatch_count: int,
) -> dict[str, Any]:
    included = targets.loc[targets.get(ORDINAL_TARGET_DEFAULT_INCLUDE_COLUMN, 0) == 1].copy()
    summary = {
        "contract_version": FIRST_SCOPE_ORDINAL_TARGET_ARTIFACT_VERSION,
        "artifact_paths": {
            "dataset_path": "data/modeling/encounter_medication_dataset_v1_65plus_first_scope.parquet",
            "clinician_review_snapshot_path": "data/labels/clinician_review_labels_v1_phase5.parquet",
            "splits_path": "data/modeling/splits_v1_65plus_first_scope.parquet",
        },
        "target_contract": {
            "label_source_name": ORDINAL_TARGET_SOURCE_NAME,
            "priority_semantic_family": PRIORITY_SCORE_SEMANTIC_FAMILY,
            "ordered_levels": list(PRIORITY_LEVELS),
            "default_training_filter_expression": (
                "target__default_training_inclusion_flag == 1 "
                "(requires latest-active clinician ordinal level, review_status == reviewed, "
                "and feature availability at review time)"
            ),
        },
        "latest_active_review_row_count": total_latest_review_rows,
        "prepared_target_row_count": int(len(targets)),
        "default_training_included_row_count": int(len(included)),
        "default_training_excluded_row_count": int(len(targets) - len(included)),
        "matched_to_dataset_row_count": int(len(targets)),
        "score_level_mismatch_row_count": score_level_mismatch_count,
        "numeric_score_row_count": int(
            pd.to_numeric(
                targets.get("target__numeric_score_available_flag", pd.Series(dtype="Int64")),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .sum()
        ),
        "numeric_score_coverage_rate": _safe_rate(
            int(
                pd.to_numeric(
                    targets.get("target__numeric_score_available_flag", pd.Series(dtype="Int64")),
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
                .sum()
            ),
            len(targets),
        ),
        "benchmark_available_row_count": int(targets["benchmark__current_rule_score"].notna().sum())
        if "benchmark__current_rule_score" in targets.columns
        else 0,
        "benchmark_available_rate": _safe_rate(
            int(targets["benchmark__current_rule_score"].notna().sum())
            if "benchmark__current_rule_score" in targets.columns
            else 0,
            len(targets),
        ),
        "review_status_frequencies": _value_count_dict(
            targets.get("label__clinician_review_status", pd.Series(dtype="object"))
        ),
        "priority_level_frequencies_all": _priority_level_frequency_dict(
            targets.get(ORDINAL_TARGET_LEVEL_COLUMN, pd.Series(dtype="string"))
        ),
        "priority_level_frequencies_default_included": _priority_level_frequency_dict(
            included.get(ORDINAL_TARGET_LEVEL_COLUMN, pd.Series(dtype="string"))
        ),
        "medication_class_frequencies": _value_count_dict(
            targets.get(
                "benchmark__medication_class_only_medication_class_standardized",
                pd.Series(dtype="object"),
            )
        ),
        "split_partition_frequencies_all": _split_count_dict(
            targets.get("split_partition", pd.Series(dtype="object"))
        ),
        "split_partition_frequencies_default_included": _split_count_dict(
            included.get("split_partition", pd.Series(dtype="object"))
        ),
        "default_training_exclusion_reason_frequencies": _value_count_dict(
            targets.get(
                ORDINAL_TARGET_DEFAULT_EXCLUSION_REASON_COLUMN,
                pd.Series(dtype="object"),
            )
        ),
        "unique_subject_count": int(targets["subject_id"].nunique(dropna=True))
        if "subject_id" in targets.columns
        else 0,
        "unique_encounter_count": int(targets["encounter_id"].nunique(dropna=True))
        if "encounter_id" in targets.columns
        else 0,
    }
    return summary


def _priority_level_frequency_dict(level_series: pd.Series) -> dict[str, int]:
    counts = {}
    normalized = level_series.astype("string")
    for level in PRIORITY_LEVELS:
        counts[level] = int(normalized.eq(level).sum())
    return counts


def _value_count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    normalized = series.dropna().astype(str)
    return {
        str(key): int(value)
        for key, value in normalized.value_counts(dropna=False).sort_index().to_dict().items()
    }


def _split_count_dict(series: pd.Series) -> dict[str, int]:
    counts = {partition.value: 0 for partition in SplitPartition}
    counts["unassigned"] = 0
    if series.empty:
        return counts
    for value in series.tolist():
        if value is None or pd.isna(value):
            counts["unassigned"] += 1
            continue
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _normalize_reason_tags(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    if isinstance(value, (tuple, set)):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        normalized = value.tolist()
        if isinstance(normalized, list):
            return [str(item).strip().lower() for item in normalized if str(item).strip()]
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except TypeError:
        pass
    return [str(value).strip().lower()] if str(value).strip() else []


def _build_modeling_row_id(row: pd.Series) -> str:
    parts: list[str] = []
    for column_name in DATASET_KEY_COLUMNS:
        value = row[column_name]
        if isinstance(value, pd.Timestamp):
            value_text = value.strftime("%Y-%m-%d %H:%M:%S")
        else:
            value_text = str(value)
        parts.append(f"{column_name}={value_text}")
    return "|".join(parts)


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


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
