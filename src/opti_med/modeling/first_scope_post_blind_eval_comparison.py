"""Post-session comparison of the ordinal model against final blind clinician reviews."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from opti_med.scoring.priority_levels import PRIORITY_LEVELS, build_priority_comparison


FIRST_SCOPE_POST_BLIND_EVAL_COMPARISON_ARTIFACT_VERSION = (
    "first_scope_post_blind_eval_comparison_v1_65plus_first_scope.v1"
)
BLIND_REVIEW_SOURCE = "blind_eval_slice_ui_phase8"


@dataclass(frozen=True, slots=True)
class FirstScopePostBlindEvalComparisonResult:
    """Rowwise post-blind comparison artifacts."""

    rowwise_comparison: pd.DataFrame
    summary: dict[str, Any]
    report_markdown: str


def build_first_scope_post_blind_eval_comparison(
    *,
    blind_eval_slice: pd.DataFrame,
    latest_clinician_reviews: pd.DataFrame,
    scored_universe: pd.DataFrame,
) -> FirstScopePostBlindEvalComparisonResult:
    """Compare the final blind-session clinician labels against model predictions."""
    required_blind_columns = [
        "modeling__row_id",
        "subject_id",
        "encounter_id",
        "medication_standardized",
        "review_timestamp",
        "medication_class_standardized",
        "queue_rank",
    ]
    required_review_columns = [
        "modeling__row_id",
        "subject_id",
        "encounter_id",
        "medication_standardized",
        "review_timestamp",
        "benchmark__medication_class_only_medication_class_standardized",
        "review_submission_id",
        "review_submission_timestamp",
        "review_version",
        "reviewer_id",
        "label__clinician_priority_level",
        "label__clinician_review_status",
        "review_provenance_json",
    ]
    required_scored_columns = [
        "modeling__row_id",
        "prediction__ordinal_priority_level",
        "prediction__probability_low",
        "prediction__probability_medium",
        "prediction__probability_high",
        "prediction__medium_high_margin",
        "benchmark__current_rule_score_level",
    ]
    _require_columns(blind_eval_slice, required_blind_columns, frame_name="blind_eval_slice")
    _require_columns(
        latest_clinician_reviews,
        required_review_columns,
        frame_name="latest_clinician_reviews",
    )
    _require_columns(scored_universe, required_scored_columns, frame_name="scored_universe")

    blind_frame = blind_eval_slice.loc[:, required_blind_columns].drop_duplicates(
        subset=["modeling__row_id"]
    ).copy()
    blind_frame = blind_frame.sort_values(["queue_rank", "modeling__row_id"], kind="mergesort")

    latest = latest_clinician_reviews.copy()
    latest["blind_eval_submission_source"] = latest["review_provenance_json"].map(
        lambda payload: (payload or {}).get("submission_source")
        if isinstance(payload, dict)
        else None
    )
    latest_blind = latest.loc[
        latest["blind_eval_submission_source"].astype(str) == BLIND_REVIEW_SOURCE
    ].copy()
    latest_blind = latest_blind.loc[
        :,
        [
            "modeling__row_id",
            "subject_id",
            "encounter_id",
            "medication_standardized",
            "review_timestamp",
            "benchmark__medication_class_only_medication_class_standardized",
            "review_submission_id",
            "review_submission_timestamp",
            "review_version",
            "reviewer_id",
            "label__clinician_priority_level",
            "label__clinician_review_status",
        ],
    ].drop_duplicates(subset=["modeling__row_id"], keep="last")

    scored = scored_universe.loc[:, required_scored_columns].drop_duplicates(
        subset=["modeling__row_id"]
    )
    comparison_frame = latest_blind.merge(
        scored,
        how="inner",
        on="modeling__row_id",
        validate="one_to_one",
    ).merge(
        blind_frame,
        how="left",
        on="modeling__row_id",
        validate="one_to_one",
        suffixes=("", "_blind_slice"),
    )
    comparison_frame["medication_class_standardized"] = comparison_frame[
        "medication_class_standardized"
    ].fillna(
        comparison_frame[
            "benchmark__medication_class_only_medication_class_standardized"
        ]
    )
    comparison_frame["blind_eval_completed_flag"] = comparison_frame[
        "review_submission_id"
    ].notna().astype(int)
    comparison_frame["evaluation_eligible_flag"] = (
        comparison_frame["blind_eval_completed_flag"].eq(1)
        & comparison_frame["label__clinician_priority_level"].notna()
        & comparison_frame["prediction__ordinal_priority_level"].notna()
    ).astype(int)

    evaluable = comparison_frame.loc[
        comparison_frame["evaluation_eligible_flag"].eq(1)
    ].copy()
    key_frame = evaluable.loc[
        :,
        [
            "modeling__row_id",
            "subject_id",
            "encounter_id",
            "medication_standardized",
            "review_timestamp",
            "medication_class_standardized",
            "queue_rank",
            "review_submission_id",
            "review_submission_timestamp",
            "review_version",
            "reviewer_id",
            "label__clinician_review_status",
            "prediction__probability_low",
            "prediction__probability_medium",
            "prediction__probability_high",
            "prediction__medium_high_margin",
            "benchmark__current_rule_score_level",
        ],
    ].copy()
    comparison = build_priority_comparison(
        key_frame=key_frame,
        left_source_name="clinician_blind_review",
        right_source_name="ordinal_model",
        left_level_series=evaluable["label__clinician_priority_level"],
        right_level_series=evaluable["prediction__ordinal_priority_level"],
    )
    rowwise = comparison.rowwise.copy()
    rowwise["artifact_version"] = FIRST_SCOPE_POST_BLIND_EVAL_COMPARISON_ARTIFACT_VERSION
    rowwise["blind_eval_completed_flag"] = 1
    rowwise = rowwise.sort_values(["queue_rank", "modeling__row_id"], kind="mergesort")

    summary = {
        "contract_version": FIRST_SCOPE_POST_BLIND_EVAL_COMPARISON_ARTIFACT_VERSION,
        "candidate_row_count": int(len(blind_frame)),
        "blind_session_latest_review_row_count": int(len(latest_blind)),
        "blind_slice_overlap_row_count": int(comparison_frame["queue_rank"].notna().sum()),
        "completed_blind_review_row_count": int(len(comparison_frame)),
        "pending_blind_review_row_count": max(int(len(blind_frame) - len(comparison_frame)), 0),
        "evaluation_eligible_row_count": int(len(evaluable)),
        "clinician_review_status_frequencies": _string_frequency(
            comparison_frame["label__clinician_review_status"]
        ),
        "clinician_level_frequencies": _string_frequency(
            evaluable["label__clinician_priority_level"]
        ),
        "model_level_frequencies": _string_frequency(
            evaluable["prediction__ordinal_priority_level"]
        ),
        "reviewed_rows_by_medication_class": _string_frequency(
            evaluable["medication_class_standardized"]
        ),
        "distinct_subject_count": int(evaluable["subject_id"].nunique(dropna=True)),
        "distinct_encounter_count": int(evaluable["encounter_id"].nunique(dropna=True)),
        "model_vs_clinician_level_comparison": comparison.summary,
        "one_vs_rest_confusion_by_level": _build_one_vs_rest_confusion(rowwise),
        "agreement_by_medication_class": _build_group_agreement(
            rowwise=rowwise, group_column="medication_class_standardized"
        ),
        "top_disagreements": _top_disagreements(rowwise),
        "notes": [
            "Clinician blind-review levels are treated as the reference label source for this post-session comparison.",
            "Same-level agreement is the primary clinical comparison lens. There is no exact numeric model score to compare against clinician optional numeric scores.",
            "True positive / true negative / false positive / false negative counts are reported one-vs-rest for each ordinal level.",
            "benchmark__current_rule_score_level is not populated on the current blind-evaluation rows, so no rule comparison is available here.",
            "The evaluation cohort is anchored on saved blind-session latest reviews. The current blind-slice artifact is used only as optional queue metadata because the slice was regenerated after the earlier queue-instability bug.",
        ],
    }
    report_markdown = build_first_scope_post_blind_eval_comparison_report(summary=summary)
    return FirstScopePostBlindEvalComparisonResult(
        rowwise_comparison=rowwise.reset_index(drop=True),
        summary=summary,
        report_markdown=report_markdown,
    )


def write_first_scope_post_blind_eval_comparison_artifacts(
    *,
    result: FirstScopePostBlindEvalComparisonResult,
    output_root: Path,
    report_output: Path | None = None,
) -> dict[str, Path]:
    """Persist the post-blind comparison artifacts."""
    output_root.mkdir(parents=True, exist_ok=True)
    rowwise_path = output_root / "rowwise_comparison.parquet"
    summary_path = output_root / "summary.json"
    report_path = output_root / "comparison_report.md"
    result.rowwise_comparison.to_parquet(rowwise_path, index=False)
    summary_path.write_text(
        json.dumps(_json_ready(result.summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(result.report_markdown, encoding="utf-8")
    artifact_paths = {
        "rowwise_comparison": rowwise_path,
        "summary": summary_path,
        "report": report_path,
    }
    if report_output is not None:
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(result.report_markdown, encoding="utf-8")
        artifact_paths["report_copy"] = report_output
    return artifact_paths


def build_first_scope_post_blind_eval_comparison_report(*, summary: dict[str, Any]) -> str:
    """Render a compact markdown comparison report."""
    comparison = summary["model_vs_clinician_level_comparison"]
    confusion = comparison["ordinal_confusion_matrix"]
    lines = [
        "# Post-Blind Clinician vs Model Comparison",
        "",
        "This artifact compares the final blind-session clinician-reviewed ordinal labels against the current ordinal model predictions.",
        "",
        "## Coverage",
        "",
        f"- current blind-slice rows: {summary['candidate_row_count']:,}",
        f"- saved blind-session latest reviews: {summary['blind_session_latest_review_row_count']:,}",
        f"- blind-slice overlap with saved session rows: {summary['blind_slice_overlap_row_count']:,}",
        f"- completed blind reviews: {summary['completed_blind_review_row_count']:,}",
        f"- pending blind reviews: {summary['pending_blind_review_row_count']:,}",
        f"- evaluation-eligible rows: {summary['evaluation_eligible_row_count']:,}",
        f"- distinct subjects: {summary['distinct_subject_count']:,}",
        f"- distinct encounters: {summary['distinct_encounter_count']:,}",
        "",
        "## Headline Agreement",
        "",
        f"- same-level agreement: {comparison['same_level_agreement_count']:,} / {comparison['level_comparable_row_count']:,} ({_percent(comparison['same_level_agreement_rate'])})",
        f"- off-by-one disagreements: {comparison['off_by_one_level_count']:,} ({_percent(comparison['off_by_one_level_rate'])})",
        f"- severe disagreements: {comparison['severe_disagreement_count']:,} ({_percent(comparison['severe_disagreement_rate'])})",
        f"- weighted quadratic kappa: {_format_optional_float(comparison['weighted_kappa_quadratic'])}",
        "",
        "## Clinician Level Mix",
        "",
    ]
    for level_name, count in summary["clinician_level_frequencies"].items():
        lines.append(f"- {level_name}: {count:,}")
    lines.extend(
        [
            "",
            "## Model Level Mix",
            "",
        ]
    )
    for level_name, count in summary["model_level_frequencies"].items():
        lines.append(f"- {level_name}: {count:,}")
    lines.extend(
        [
            "",
            "## Ordinal Confusion Matrix",
            "",
            "Rows are final blind clinician labels. Columns are model-predicted levels.",
            "",
            "| clinician \\/ model | low | medium | high |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for clinician_level in PRIORITY_LEVELS:
        row_counts = confusion[clinician_level]
        lines.append(
            f"| {clinician_level} | {row_counts['low']:,} | {row_counts['medium']:,} | {row_counts['high']:,} |"
        )
    lines.extend(
        [
            "",
            "## One-vs-Rest TP/TN/FP/FN",
            "",
            "These counts treat the clinician blind label as ground truth and evaluate one ordinal level at a time against all other levels.",
            "",
        ]
    )
    for level_name in PRIORITY_LEVELS:
        payload = summary["one_vs_rest_confusion_by_level"][level_name]
        lines.extend(
            [
                f"### {level_name.capitalize()}",
                "",
                f"- TP: {payload['true_positive_count']:,}",
                f"- TN: {payload['true_negative_count']:,}",
                f"- FP: {payload['false_positive_count']:,}",
                f"- FN: {payload['false_negative_count']:,}",
                f"- precision: {_percent(payload['precision'])}",
                f"- recall: {_percent(payload['recall'])}",
                f"- specificity: {_percent(payload['specificity'])}",
                "",
            ]
        )
    lines.extend(
        [
            "## Agreement by Medication Class",
            "",
        ]
    )
    for class_name, payload in summary["agreement_by_medication_class"].items():
        lines.append(
            f"- {class_name}: {payload['same_level_agreement_count']:,} / {payload['row_count']:,} ({_percent(payload['same_level_agreement_rate'])})"
        )
    lines.extend(
        [
            "",
            "## Top Disagreements",
            "",
        ]
    )
    if summary["top_disagreements"]:
        for item in summary["top_disagreements"]:
            lines.append(
                "- "
                + (
                    f"rank {item['queue_rank']}: patient {item['subject_id']} / {item['encounter_id']} / "
                    f"{item['medication_standardized']} / clinician {item['clinician_level']} / "
                    f"model {item['model_level']} / level distance {item['level_distance']}"
                )
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    for note in summary["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _build_one_vs_rest_confusion(rowwise: pd.DataFrame) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    if rowwise.empty:
        for level_name in PRIORITY_LEVELS:
            output[level_name] = {
                "true_positive_count": 0,
                "true_negative_count": 0,
                "false_positive_count": 0,
                "false_negative_count": 0,
                "precision": None,
                "recall": None,
                "specificity": None,
            }
        return output

    truth = rowwise["comparison__left_level"].astype(str)
    pred = rowwise["comparison__right_level"].astype(str)
    for level_name in PRIORITY_LEVELS:
        tp = int(((truth == level_name) & (pred == level_name)).sum())
        tn = int(((truth != level_name) & (pred != level_name)).sum())
        fp = int(((truth != level_name) & (pred == level_name)).sum())
        fn = int(((truth == level_name) & (pred != level_name)).sum())
        output[level_name] = {
            "true_positive_count": tp,
            "true_negative_count": tn,
            "false_positive_count": fp,
            "false_negative_count": fn,
            "precision": (tp / (tp + fp)) if (tp + fp) else None,
            "recall": (tp / (tp + fn)) if (tp + fn) else None,
            "specificity": (tn / (tn + fp)) if (tn + fp) else None,
        }
    return output


def _build_group_agreement(*, rowwise: pd.DataFrame, group_column: str) -> dict[str, dict[str, Any]]:
    if rowwise.empty:
        return {}
    groups: dict[str, dict[str, Any]] = {}
    for group_value, frame in rowwise.groupby(group_column, dropna=False):
        group_name = str(group_value) if pd.notna(group_value) else "unknown"
        row_count = int(len(frame))
        same_level_count = int(frame["comparison__same_level_match_flag"].fillna(False).astype(bool).sum())
        groups[group_name] = {
            "row_count": row_count,
            "same_level_agreement_count": same_level_count,
            "same_level_agreement_rate": (same_level_count / row_count) if row_count else None,
        }
    return dict(sorted(groups.items()))


def _top_disagreements(rowwise: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    if rowwise.empty:
        return []
    disagreements = rowwise.loc[
        rowwise["comparison__same_level_match_flag"].fillna(False).astype(bool).eq(False)
    ].copy()
    if disagreements.empty:
        return []
    disagreements = disagreements.sort_values(
        ["comparison__level_distance", "prediction__medium_high_margin", "queue_rank"],
        ascending=[False, True, True],
        na_position="last",
        kind="mergesort",
    ).head(limit)
    payload: list[dict[str, Any]] = []
    for record in disagreements.to_dict(orient="records"):
        payload.append(
            {
                "queue_rank": int(record.get("queue_rank") or 0),
                "subject_id": int(record["subject_id"]),
                "encounter_id": str(record["encounter_id"]),
                "medication_standardized": str(record["medication_standardized"]),
                "clinician_level": str(record.get("comparison__left_level") or ""),
                "model_level": str(record.get("comparison__right_level") or ""),
                "level_distance": int(record.get("comparison__level_distance") or 0),
            }
        )
    return payload


def _string_frequency(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    counts = (
        series.fillna("missing")
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    return {str(key): int(value) for key, value in counts.items()}


def _require_columns(frame: pd.DataFrame, columns: list[str], *, frame_name: str) -> None:
    missing = [column_name for column_name in columns if column_name not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {', '.join(missing)}")


def _percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value
