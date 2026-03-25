"""Shared priority score level mapping and ordinal comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


PRIORITY_LEVEL_LOW = "low"
PRIORITY_LEVEL_MEDIUM = "medium"
PRIORITY_LEVEL_HIGH = "high"
PRIORITY_LEVELS: tuple[str, ...] = (
    PRIORITY_LEVEL_LOW,
    PRIORITY_LEVEL_MEDIUM,
    PRIORITY_LEVEL_HIGH,
)
PRIORITY_LEVEL_BUCKETS: tuple[dict[str, object], ...] = (
    {
        "level": PRIORITY_LEVEL_LOW,
        "min_inclusive": 0.0,
        "max_exclusive": 4.0,
        "display_range": "0-3",
    },
    {
        "level": PRIORITY_LEVEL_MEDIUM,
        "min_inclusive": 4.0,
        "max_exclusive": 7.0,
        "display_range": "4-6",
    },
    {
        "level": PRIORITY_LEVEL_HIGH,
        "min_inclusive": 7.0,
        "max_inclusive": 10.0,
        "display_range": "7-10",
    },
)
PRIORITY_LEVEL_ORDER = {
    PRIORITY_LEVEL_LOW: 0,
    PRIORITY_LEVEL_MEDIUM: 1,
    PRIORITY_LEVEL_HIGH: 2,
}
PRIORITY_SCORE_MIN = 0.0
PRIORITY_SCORE_MAX = 10.0
PRIORITY_SCORE_SEMANTIC_FAMILY = "canonical_priority_score_0_to_10"
PRIORITY_COMPARISON_STATUSES: tuple[str, ...] = (
    "unavailable",
    "level_only_available",
    "level_and_score_available",
)


@dataclass(frozen=True, slots=True)
class PriorityComparisonResult:
    """Rowwise and summary outputs for one ordinal priority comparison."""

    rowwise: pd.DataFrame
    summary: dict[str, Any]


def coerce_priority_score_value(score: object) -> float | None:
    """Coerce one priority score value and validate the 0-to-10 clinical range."""
    if score is None or pd.isna(score):
        return None
    numeric_score = pd.to_numeric(pd.Series([score]), errors="coerce").iloc[0]
    if pd.isna(numeric_score):
        return None
    numeric_score = float(numeric_score)
    if numeric_score < PRIORITY_SCORE_MIN or numeric_score > PRIORITY_SCORE_MAX:
        raise ValueError(
            "Priority scores must stay within the canonical 0-to-10 range."
        )
    return numeric_score


def score_to_priority_level(score: object) -> str | None:
    """Map a canonical 0-to-10 priority score to low/medium/high."""
    numeric_score = coerce_priority_score_value(score)
    if numeric_score is None:
        return None
    for bucket in PRIORITY_LEVEL_BUCKETS:
        max_exclusive = bucket.get("max_exclusive")
        if max_exclusive is not None and numeric_score < float(max_exclusive):
            return str(bucket["level"])
        max_inclusive = bucket.get("max_inclusive")
        if max_inclusive is not None and numeric_score <= float(max_inclusive):
            return str(bucket["level"])
    raise ValueError("Priority score bucket mapping is misconfigured.")


def normalize_priority_level(level: object) -> str | None:
    """Normalize a low/medium/high level value or return None when missing."""
    if level is None or pd.isna(level):
        return None
    normalized = str(level).strip().lower()
    if not normalized:
        return None
    if normalized not in PRIORITY_LEVEL_ORDER:
        raise ValueError(
            "Priority levels must be one of: low, medium, high."
        )
    return normalized


def derive_priority_level_series(score_series: pd.Series) -> pd.Series:
    """Derive stable low/medium/high levels from a canonical numeric score series."""
    return score_series.apply(score_to_priority_level).astype("string")


def build_priority_level_contract_payload() -> dict[str, Any]:
    """Return the canonical, reusable level-mapping contract payload."""
    return {
        "semantic_family": PRIORITY_SCORE_SEMANTIC_FAMILY,
        "score_min": PRIORITY_SCORE_MIN,
        "score_max": PRIORITY_SCORE_MAX,
        "level_order": list(PRIORITY_LEVELS),
        "ordered_buckets": [
            {
                "level": str(bucket["level"]),
                "display_range": str(bucket["display_range"]),
            }
            for bucket in PRIORITY_LEVEL_BUCKETS
        ],
        "mapping": {
            str(bucket["level"]): str(bucket["display_range"])
            for bucket in PRIORITY_LEVEL_BUCKETS
        },
        "float_bucketing_policy": "[0.0, 4.0) => low; [4.0, 7.0) => medium; [7.0, 10.0] => high",
        "primary_clinical_comparison_lens": "same_level_agreement",
        "exact_score_comparison_role": "secondary_diagnostic_only",
    }


def resolve_priority_level_series(
    *,
    score_series: pd.Series | None = None,
    level_series: pd.Series | None = None,
    score_name: str,
    level_name: str,
) -> pd.Series:
    """Return normalized levels, filling missing level values from the canonical score mapping."""
    if score_series is None and level_series is None:
        raise ValueError("A priority level series requires a score series or a level series.")

    if level_series is not None:
        resolved = level_series.reset_index(drop=True).apply(normalize_priority_level).astype(
            "string"
        )
    elif score_series is not None:
        resolved = pd.Series(pd.NA, index=score_series.reset_index(drop=True).index, dtype="string")
    else:
        raise ValueError("A priority level series requires a score series or a level series.")

    if score_series is None:
        return resolved

    coerced_score_series = score_series.reset_index(drop=True).apply(coerce_priority_score_value)
    derived_levels = coerced_score_series.apply(score_to_priority_level).astype("string")
    if level_series is not None:
        validate_score_level_alignment(
            score_series=coerced_score_series,
            level_series=resolved,
            score_name=score_name,
            level_name=level_name,
        )
    return resolved.fillna(derived_levels).astype("string")


def validate_score_level_alignment(
    *,
    score_series: pd.Series,
    level_series: pd.Series,
    score_name: str,
    level_name: str,
) -> None:
    """Ensure a provided level column matches the canonical score-to-level mapping."""
    expected_levels = score_series.apply(score_to_priority_level)
    normalized_levels = level_series.apply(normalize_priority_level)
    comparable_mask = expected_levels.notna() & normalized_levels.notna()
    if not expected_levels.loc[comparable_mask].eq(normalized_levels.loc[comparable_mask]).all():
        raise ValueError(
            f"{level_name} must stay aligned with the canonical level mapping for {score_name}."
        )


def build_priority_comparison(
    *,
    key_frame: pd.DataFrame,
    left_source_name: str,
    right_source_name: str,
    left_score_series: pd.Series | None = None,
    right_score_series: pd.Series | None = None,
    left_level_series: pd.Series | None = None,
    right_level_series: pd.Series | None = None,
) -> PriorityComparisonResult:
    """Build rowwise and summary ordinal comparison outputs for two score sources."""
    row_count = len(key_frame)
    if row_count == 0:
        empty = key_frame.copy().reset_index(drop=True)
        for column_name in _comparison_detail_columns():
            empty[column_name] = pd.Series(dtype="object")
        return PriorityComparisonResult(
            rowwise=empty,
            summary={
                "left_source_name": left_source_name,
                "right_source_name": right_source_name,
                "status": "unavailable",
                "status_reason": "no_rows_available",
                "level_comparable_row_count": 0,
                "score_comparable_row_count": 0,
                "same_level_agreement_count": 0,
                "same_level_agreement_rate": None,
                "exact_score_agreement_count": 0,
                "exact_score_agreement_rate": None,
                "mean_absolute_score_error": None,
                "median_absolute_score_error": None,
                "off_by_one_level_count": 0,
                "off_by_one_level_rate": None,
                "severe_disagreement_count": 0,
                "severe_disagreement_rate": None,
                "weighted_kappa_quadratic": None,
                "ordinal_confusion_matrix": _empty_ordinal_confusion_matrix(),
                "primary_headline_metric": "same_level_agreement_rate",
                "warnings": ["No rows available for comparison."],
            },
        )

    if left_score_series is None and left_level_series is None:
        raise ValueError("Left comparison source requires a score or level series.")
    if right_score_series is None and right_level_series is None:
        raise ValueError("Right comparison source requires a score or level series.")

    left_score = (
        left_score_series.reset_index(drop=True).apply(coerce_priority_score_value)
        if left_score_series is not None
        else pd.Series([None] * row_count, dtype="object")
    )
    right_score = (
        right_score_series.reset_index(drop=True).apply(coerce_priority_score_value)
        if right_score_series is not None
        else pd.Series([None] * row_count, dtype="object")
    )
    left_level = resolve_priority_level_series(
        score_series=left_score,
        level_series=left_level_series,
        score_name=left_source_name,
        level_name=f"{left_source_name}_level",
    )
    right_level = resolve_priority_level_series(
        score_series=right_score,
        level_series=right_level_series,
        score_name=right_source_name,
        level_name=f"{right_source_name}_level",
    )

    left_level_available = left_level.notna()
    right_level_available = right_level.notna()
    left_score_available = left_score.notna()
    right_score_available = right_score.notna()
    level_comparable_mask = left_level_available & right_level_available
    score_comparable_mask = left_score_available & right_score_available

    left_level_index = left_level.map(PRIORITY_LEVEL_ORDER)
    right_level_index = right_level.map(PRIORITY_LEVEL_ORDER)
    level_distance = pd.Series([pd.NA] * row_count, dtype="Int64")
    level_distance.loc[level_comparable_mask] = (
        left_level_index.loc[level_comparable_mask]
        .sub(right_level_index.loc[level_comparable_mask])
        .abs()
        .astype("Int64")
    )
    same_level_match = pd.Series([pd.NA] * row_count, dtype="boolean")
    same_level_match.loc[level_comparable_mask] = left_level.loc[level_comparable_mask].eq(
        right_level.loc[level_comparable_mask]
    ).astype("boolean")
    off_by_one = pd.Series([pd.NA] * row_count, dtype="boolean")
    off_by_one.loc[level_comparable_mask] = level_distance.loc[level_comparable_mask].eq(1)
    severe_disagreement = pd.Series([pd.NA] * row_count, dtype="boolean")
    severe_disagreement.loc[level_comparable_mask] = level_distance.loc[level_comparable_mask].eq(2)

    exact_score_match = pd.Series([pd.NA] * row_count, dtype="boolean")
    exact_score_match.loc[score_comparable_mask] = (
        left_score.loc[score_comparable_mask].astype(float).round(12)
        == right_score.loc[score_comparable_mask].astype(float).round(12)
    )
    absolute_score_error = pd.Series([np.nan] * row_count, dtype="float64")
    absolute_score_error.loc[score_comparable_mask] = (
        left_score.loc[score_comparable_mask].astype(float)
        .sub(right_score.loc[score_comparable_mask].astype(float))
        .abs()
    )

    comparison_status = pd.Series(["unavailable"] * row_count, dtype="string")
    comparison_status.loc[level_comparable_mask] = "level_only_available"
    comparison_status.loc[level_comparable_mask & score_comparable_mask] = (
        "level_and_score_available"
    )

    rowwise = key_frame.copy().reset_index(drop=True)
    rowwise["comparison__left_source_name"] = left_source_name
    rowwise["comparison__right_source_name"] = right_source_name
    rowwise["comparison__left_score_value"] = left_score
    rowwise["comparison__right_score_value"] = right_score
    rowwise["comparison__left_level"] = left_level.astype("string")
    rowwise["comparison__right_level"] = right_level.astype("string")
    rowwise["comparison__status"] = comparison_status
    rowwise["comparison__same_level_match_flag"] = same_level_match
    rowwise["comparison__exact_score_match_flag"] = exact_score_match
    rowwise["comparison__absolute_score_error"] = absolute_score_error
    rowwise["comparison__level_distance"] = level_distance
    rowwise["comparison__off_by_one_level_flag"] = off_by_one
    rowwise["comparison__severe_disagreement_flag"] = severe_disagreement
    rowwise = rowwise.sort_values(list(key_frame.columns), na_position="last").reset_index(drop=True)

    comparable_level_rows = int(level_comparable_mask.sum())
    comparable_score_rows = int(score_comparable_mask.sum())
    warnings: list[str] = []
    if comparable_level_rows == 0:
        warnings.append("No rows had two comparable low/medium/high level sources.")
    if comparable_score_rows == 0:
        warnings.append("No rows had two comparable canonical 0-to-10 numeric score sources.")
    weighted_kappa = None
    if comparable_level_rows >= 2:
        left_codes = left_level_index.loc[level_comparable_mask].astype(int).tolist()
        right_codes = right_level_index.loc[level_comparable_mask].astype(int).tolist()
        if len(set(left_codes + right_codes)) >= 2:
            weighted_kappa = float(
                cohen_kappa_score(left_codes, right_codes, weights="quadratic")
            )
        else:
            warnings.append("Weighted kappa is not informative when only one ordinal level is observed.")
    else:
        warnings.append("Weighted kappa requires at least two comparable level rows.")

    confusion_matrix = _empty_ordinal_confusion_matrix()
    if comparable_level_rows > 0:
        for left_level_name in PRIORITY_LEVELS:
            left_mask = rowwise["comparison__left_level"] == left_level_name
            for right_level_name in PRIORITY_LEVELS:
                count = int(
                    (
                        left_mask
                        & rowwise["comparison__right_level"].eq(right_level_name)
                    ).sum()
                )
                confusion_matrix[left_level_name][right_level_name] = count

    same_level_count = int(same_level_match.fillna(False).astype(bool).sum())
    exact_score_count = int(exact_score_match.fillna(False).astype(bool).sum())
    off_by_one_count = int(off_by_one.fillna(False).astype(bool).sum())
    severe_count = int(severe_disagreement.fillna(False).astype(bool).sum())

    status = "available" if comparable_level_rows > 0 else "unavailable"
    status_reason = (
        "level-aware comparison is available."
        if comparable_level_rows > 0
        else "no rows had two comparable ordinal level sources."
    )
    return PriorityComparisonResult(
        rowwise=rowwise,
        summary={
            "left_source_name": left_source_name,
            "right_source_name": right_source_name,
            "status": status,
            "status_reason": status_reason,
            "level_comparable_row_count": comparable_level_rows,
            "score_comparable_row_count": comparable_score_rows,
            "same_level_agreement_count": same_level_count,
            "same_level_agreement_rate": (
                same_level_count / comparable_level_rows
                if comparable_level_rows
                else None
            ),
            "exact_score_agreement_count": exact_score_count,
            "exact_score_agreement_rate": (
                exact_score_count / comparable_score_rows
                if comparable_score_rows
                else None
            ),
            "mean_absolute_score_error": (
                float(absolute_score_error.loc[score_comparable_mask].mean())
                if comparable_score_rows
                else None
            ),
            "median_absolute_score_error": (
                float(absolute_score_error.loc[score_comparable_mask].median())
                if comparable_score_rows
                else None
            ),
            "off_by_one_level_count": off_by_one_count,
            "off_by_one_level_rate": (
                off_by_one_count / comparable_level_rows
                if comparable_level_rows
                else None
            ),
            "severe_disagreement_count": severe_count,
            "severe_disagreement_rate": (
                severe_count / comparable_level_rows
                if comparable_level_rows
                else None
            ),
            "weighted_kappa_quadratic": weighted_kappa,
            "ordinal_confusion_matrix": confusion_matrix,
            "primary_headline_metric": "same_level_agreement_rate",
            "warnings": sorted(set(warnings)),
        },
    )


def _empty_ordinal_confusion_matrix() -> dict[str, dict[str, int]]:
    """Return a stable empty low/medium/high confusion matrix."""
    return {
        left_level: {right_level: 0 for right_level in PRIORITY_LEVELS}
        for left_level in PRIORITY_LEVELS
    }


def _comparison_detail_columns() -> list[str]:
    """Return the stable rowwise comparison columns."""
    return [
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
