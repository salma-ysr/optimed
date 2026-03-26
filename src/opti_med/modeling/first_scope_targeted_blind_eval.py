"""Targeted blind clinician evaluation slice using real ordinal model predictions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from opti_med.api.clinician_reviews import build_clinician_review_queue
from opti_med.data_access.exceptions import DataLoadError
from opti_med.modeling.first_scope_ordinal_baseline import (
    build_first_scope_ordinal_scored_universe,
)


TARGETED_BLIND_EVAL_CONTRACT_VERSION = (
    "clinician_review_targeted_blind_eval_slice_v1_65plus_first_scope.v1"
)
TARGETED_BLIND_EVAL_BANDS = (
    "model_rule_disagreement",
    "uncertain_existing_review",
    "medium_high_boundary",
    "high_signal_unreviewed_fallback",
)
SCORED_UNIVERSE_COLUMNS = [
    "modeling__row_id",
    "prediction__ordinal_priority_level",
    "prediction__probability_low",
    "prediction__probability_medium",
    "prediction__probability_high",
    "prediction__medium_high_margin",
]


@dataclass(frozen=True, slots=True)
class FirstScopeTargetedBlindEvalResult:
    """Full scored universe plus targeted blind evaluation slice artifacts."""

    scored_universe: pd.DataFrame
    blind_eval_slice: pd.DataFrame
    summary: dict[str, Any]
    report_markdown: str


def build_first_scope_targeted_blind_eval(
    *,
    encounter_medication_dataset: pd.DataFrame,
    reviewable_universe: pd.DataFrame,
    latest_clinician_reviews: pd.DataFrame,
    ordinal_targets: pd.DataFrame,
    random_seed: int,
    medium_high_margin_threshold: float = 0.18,
    fallback_high_probability_threshold: float = 0.60,
    fallback_minimum_candidate_rows: int = 24,
) -> FirstScopeTargetedBlindEvalResult:
    """Build a deterministic blind clinician evaluation slice from real ordinal predictions."""
    scored_universe = build_first_scope_ordinal_scored_universe(
        encounter_medication_dataset=encounter_medication_dataset,
        ordinal_targets=ordinal_targets,
        random_seed=random_seed,
    )
    review_queue = build_clinician_review_queue(
        latest_reviews=latest_clinician_reviews,
        reviewable_universe=reviewable_universe,
    )
    slice_frame = review_queue.merge(
        scored_universe.loc[:, SCORED_UNIVERSE_COLUMNS],
        how="left",
        on="modeling__row_id",
        validate="one_to_one",
    )
    if slice_frame["prediction__ordinal_priority_level"].isna().any():
        raise DataLoadError(
            "Targeted blind evaluation slice requires ordinal predictions for every reviewable row."
        )

    slice_frame["blind_eval_model_rule_disagreement_flag"] = slice_frame.apply(
        lambda row: int(
            pd.notna(row.get("benchmark__current_rule_score_level"))
            and str(row.get("prediction__ordinal_priority_level")) != str(row.get("benchmark__current_rule_score_level"))
        ),
        axis=1,
    )
    slice_frame["blind_eval_uncertain_existing_review_flag"] = slice_frame[
        "label__clinician_review_status"
    ].fillna("").astype(str).str.lower().eq("uncertain").astype(int)
    slice_frame["blind_eval_medium_high_boundary_flag"] = slice_frame.apply(
        lambda row: int(
            _medium_high_boundary_candidate(
                predicted_level=row.get("prediction__ordinal_priority_level"),
                probability_medium=row.get("prediction__probability_medium"),
                probability_high=row.get("prediction__probability_high"),
                margin_threshold=medium_high_margin_threshold,
            )
        ),
        axis=1,
    )
    slice_frame["blind_eval_high_signal_unreviewed_fallback_flag"] = 0

    initial_candidate_mask = (
        slice_frame["blind_eval_model_rule_disagreement_flag"].eq(1)
        | slice_frame["blind_eval_uncertain_existing_review_flag"].eq(1)
        | slice_frame["blind_eval_medium_high_boundary_flag"].eq(1)
    )
    if int(initial_candidate_mask.sum()) < fallback_minimum_candidate_rows:
        required_fallback_rows = fallback_minimum_candidate_rows - int(initial_candidate_mask.sum())
        fallback_mask = (
            slice_frame["current_clinician_reviewed_flag"].eq(0)
            & slice_frame["prediction__ordinal_priority_level"].eq("high")
            & pd.to_numeric(
                slice_frame["prediction__probability_high"],
                errors="coerce",
            )
            .fillna(0.0)
            .ge(fallback_high_probability_threshold)
            & ~initial_candidate_mask
        )
        fallback_candidates = slice_frame.loc[fallback_mask].copy()
        if required_fallback_rows > 0 and not fallback_candidates.empty:
            fallback_candidates = fallback_candidates.sort_values(
                [
                    "prediction__probability_high",
                    "queue_rank",
                    "subject_id",
                    "encounter_id",
                    "medication_standardized",
                    "review_timestamp",
                ],
                ascending=[False, True, True, True, True, True],
                na_position="last",
                kind="mergesort",
            ).head(required_fallback_rows)
            slice_frame.loc[
                fallback_candidates.index,
                "blind_eval_high_signal_unreviewed_fallback_flag",
            ] = 1

    slice_frame["queue_priority_reasons"] = slice_frame.apply(_blind_eval_reasons, axis=1)
    slice_frame["queue_priority_band"] = slice_frame.apply(_blind_eval_band, axis=1)
    slice_frame["queue_priority_score"] = slice_frame.apply(_blind_eval_priority_score, axis=1)
    slice_frame["needs_review_justification"] = slice_frame["queue_priority_reasons"].map(
        _blind_eval_reason_summary
    )
    candidate = slice_frame.loc[
        slice_frame["queue_priority_reasons"].map(bool)
    ].copy()
    candidate = candidate.sort_values(
        [
            "queue_priority_score",
            "current_clinician_reviewed_flag",
            "queue_rank_within_subject",
            "queue_rank_within_medication_class",
            "class_reviewed_count",
            "subject_reviewed_count",
            "prediction__probability_high",
            "prediction__probability_medium",
            "subject_id",
            "encounter_id",
            "medication_standardized",
            "review_timestamp",
        ],
        ascending=[False, True, True, True, True, True, False, False, True, True, True, False],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    candidate["queue_rank"] = range(1, len(candidate) + 1)
    candidate["targeted_blind_eval_contract_version"] = TARGETED_BLIND_EVAL_CONTRACT_VERSION

    summary = _build_targeted_blind_eval_summary(
        scored_universe=scored_universe,
        blind_eval_slice=candidate,
        fallback_minimum_candidate_rows=fallback_minimum_candidate_rows,
        medium_high_margin_threshold=medium_high_margin_threshold,
        fallback_high_probability_threshold=fallback_high_probability_threshold,
    )
    report_markdown = build_targeted_blind_eval_report(summary=summary)
    return FirstScopeTargetedBlindEvalResult(
        scored_universe=scored_universe,
        blind_eval_slice=candidate,
        summary=summary,
        report_markdown=report_markdown,
    )


def write_targeted_blind_eval_artifacts(
    *,
    result: FirstScopeTargetedBlindEvalResult,
    scored_universe_output_path: Path,
    slice_output_path: Path,
    summary_output_path: Path,
    report_output_path: Path,
) -> dict[str, Path]:
    """Persist the scored universe and targeted blind evaluation slice artifacts."""
    scored_universe_output_path.parent.mkdir(parents=True, exist_ok=True)
    slice_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    result.scored_universe.to_parquet(scored_universe_output_path, index=False)
    result.blind_eval_slice.to_parquet(slice_output_path, index=False)
    summary_payload = dict(result.summary)
    summary_payload["artifact_paths"] = {
        **summary_payload.get("artifact_paths", {}),
        "scored_universe_output_path": str(scored_universe_output_path),
        "blind_eval_slice_output_path": str(slice_output_path),
        "blind_eval_summary_output_path": str(summary_output_path),
        "blind_eval_report_output_path": str(report_output_path),
    }
    summary_output_path.write_text(
        json.dumps(_json_ready(summary_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_output_path.write_text(result.report_markdown, encoding="utf-8")
    return {
        "scored_universe": scored_universe_output_path,
        "blind_eval_slice": slice_output_path,
        "summary": summary_output_path,
        "report": report_output_path,
    }


def filter_targeted_blind_eval_slice(
    blind_eval_slice: pd.DataFrame,
    *,
    unreviewed_only: bool = False,
    medication_class: str | None = None,
    review_status: str = "all",
    subject_id: int | None = None,
    priority_band: str = "all",
) -> pd.DataFrame:
    """Apply deterministic filters to the targeted blind evaluation slice."""
    filtered = blind_eval_slice.copy()
    if filtered.empty:
        return filtered
    if unreviewed_only:
        filtered = filtered.loc[filtered["current_clinician_reviewed_flag"] == 0].copy()
    if medication_class:
        filtered = filtered.loc[
            filtered["medication_class_standardized"].astype(str).str.lower()
            == str(medication_class).strip().lower()
        ].copy()
    normalized_review_status = str(review_status or "all").strip().lower()
    if normalized_review_status == "unreviewed":
        filtered = filtered.loc[filtered["current_clinician_reviewed_flag"] == 0].copy()
    elif normalized_review_status in {"reviewed", "uncertain", "insufficient_context", "skip"}:
        filtered = filtered.loc[
            filtered["label__clinician_review_status"].fillna("").astype(str).str.lower()
            == normalized_review_status
        ].copy()
    if subject_id is not None:
        filtered = filtered.loc[
            pd.to_numeric(filtered["subject_id"], errors="coerce") == int(subject_id)
        ].copy()
    normalized_priority_band = str(priority_band or "all").strip().lower()
    if normalized_priority_band != "all":
        filtered = filtered.loc[
            filtered["queue_priority_band"].fillna("").astype(str).str.lower()
            == normalized_priority_band
        ].copy()
    return filtered.reset_index(drop=True)


def build_targeted_blind_eval_report(*, summary: dict[str, Any]) -> str:
    """Render a compact markdown report for the targeted blind evaluation slice."""
    lines = [
        "# Targeted Blind Clinician Evaluation Slice",
        "",
        "This artifact uses the real ordinal logistic model to prioritize the final clinician session. It is not built from dummy predictions.",
        "",
        "## Input Artifacts",
        "",
        f"- dataset: `{summary['artifact_paths']['dataset_path']}`",
        f"- ordinal targets: `{summary['artifact_paths']['ordinal_targets_path']}`",
        f"- clinician review snapshot: `{summary['artifact_paths']['clinician_review_snapshot_path']}`",
        f"- reviewable universe: `{summary['artifact_paths']['reviewable_universe_path']}`",
        "",
        "## Slice Logic",
        "",
        f"- contract_version: `{summary['contract_version']}`",
        f"- medium/high boundary margin threshold: `{summary['selection_logic']['medium_high_margin_threshold']:.2f}`",
        f"- fallback high-signal threshold: `{summary['selection_logic']['fallback_high_probability_threshold']:.2f}`",
        f"- minimum target candidate count before fallback: `{summary['selection_logic']['fallback_minimum_candidate_rows']}`",
        "- reviewer-facing UI hides model/rule cues and queue rationale during the blind session.",
        "",
        "## Counts",
        "",
        f"- reviewable rows scored by the real ordinal model: {summary['scored_reviewable_row_count']:,}",
        f"- blind evaluation candidates: {summary['blind_eval_candidate_row_count']:,}",
        f"- model-vs-rule disagreement candidates: {summary['model_rule_disagreement_candidate_count']:,}",
        f"- medium/high boundary candidates: {summary['medium_high_boundary_candidate_count']:,}",
        f"- uncertain-review revisit candidates: {summary['uncertain_existing_review_candidate_count']:,}",
        f"- fallback high-signal unreviewed candidates: {summary['high_signal_unreviewed_fallback_candidate_count']:,}",
        "",
        "## Composition",
        "",
        f"- unreviewed candidates: {summary['unreviewed_candidate_row_count']:,}",
        f"- already-reviewed candidates: {summary['already_reviewed_candidate_row_count']:,}",
        f"- medication classes: {summary['candidate_rows_by_medication_class']}",
        f"- subjects covered: {summary['candidate_distinct_subject_count']:,}",
        f"- encounters covered: {summary['candidate_distinct_encounter_count']:,}",
        "",
        "## Notes",
        "",
    ]
    for note in summary["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _medium_high_boundary_candidate(
    *,
    predicted_level: object,
    probability_medium: object,
    probability_high: object,
    margin_threshold: float,
) -> bool:
    normalized_level = str(predicted_level or "").strip().lower()
    medium = float(pd.to_numeric(pd.Series([probability_medium]), errors="coerce").fillna(0.0).iloc[0])
    high = float(pd.to_numeric(pd.Series([probability_high]), errors="coerce").fillna(0.0).iloc[0])
    return (
        normalized_level in {"medium", "high"}
        and abs(high - medium) <= float(margin_threshold)
    )


def _blind_eval_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if int(row.get("blind_eval_model_rule_disagreement_flag") or 0) == 1:
        reasons.append("model_rule_disagreement")
    if int(row.get("blind_eval_uncertain_existing_review_flag") or 0) == 1:
        reasons.append("uncertain_existing_review")
    if int(row.get("blind_eval_medium_high_boundary_flag") or 0) == 1:
        reasons.append("medium_high_boundary")
    if int(row.get("blind_eval_high_signal_unreviewed_fallback_flag") or 0) == 1:
        reasons.append("high_signal_unreviewed_fallback")
    return reasons


def _blind_eval_band(row: pd.Series) -> str:
    reasons = _blind_eval_reasons(row)
    if not reasons:
        return "not_selected"
    if len(reasons) == 1:
        return reasons[0]
    return "mixed_signal_candidate"


def _blind_eval_priority_score(row: pd.Series) -> int:
    score = 0
    if int(row.get("blind_eval_model_rule_disagreement_flag") or 0) == 1:
        score += 400
    if int(row.get("blind_eval_uncertain_existing_review_flag") or 0) == 1:
        score += 320
    if int(row.get("blind_eval_medium_high_boundary_flag") or 0) == 1:
        score += 240
    if int(row.get("blind_eval_high_signal_unreviewed_fallback_flag") or 0) == 1:
        score += 160
    if int(row.get("current_clinician_reviewed_flag") or 0) == 0:
        score += 20
    return score


def _blind_eval_reason_summary(reasons: list[str]) -> str:
    if not reasons:
        return ""
    labels = {
        "model_rule_disagreement": "Model and current rule disagree on the ordinal level.",
        "uncertain_existing_review": "A prior clinician review exists but remained uncertain.",
        "medium_high_boundary": "The real ordinal model is near the medium/high decision boundary.",
        "high_signal_unreviewed_fallback": "Fallback inclusion because the model assigns a strong unreviewed high-risk signal.",
    }
    return " ".join(labels.get(reason, reason) for reason in reasons)


def _build_targeted_blind_eval_summary(
    *,
    scored_universe: pd.DataFrame,
    blind_eval_slice: pd.DataFrame,
    fallback_minimum_candidate_rows: int,
    medium_high_margin_threshold: float,
    fallback_high_probability_threshold: float,
) -> dict[str, Any]:
    return {
        "contract_version": TARGETED_BLIND_EVAL_CONTRACT_VERSION,
        "artifact_paths": {
            "dataset_path": "data/modeling/encounter_medication_dataset_v1_65plus_first_scope.parquet",
            "ordinal_targets_path": "data/modeling/encounter_medication_ordinal_targets_v1_65plus_first_scope.parquet",
            "clinician_review_snapshot_path": "data/labels/clinician_review_labels_v1_phase5.parquet",
            "reviewable_universe_path": "data/labels/clinician_review_universe_v1_phase6.parquet",
        },
        "selection_logic": {
            "medium_high_margin_threshold": float(medium_high_margin_threshold),
            "fallback_high_probability_threshold": float(fallback_high_probability_threshold),
            "fallback_minimum_candidate_rows": int(fallback_minimum_candidate_rows),
            "real_model_source": "logistic_regression_multinomial",
        },
        "scored_reviewable_row_count": int(len(scored_universe)),
        "blind_eval_candidate_row_count": int(len(blind_eval_slice)),
        "model_rule_disagreement_candidate_count": int(
            blind_eval_slice["queue_priority_reasons"].map(lambda reasons: "model_rule_disagreement" in reasons).sum()
        )
        if not blind_eval_slice.empty
        else 0,
        "medium_high_boundary_candidate_count": int(
            blind_eval_slice["queue_priority_reasons"].map(lambda reasons: "medium_high_boundary" in reasons).sum()
        )
        if not blind_eval_slice.empty
        else 0,
        "uncertain_existing_review_candidate_count": int(
            blind_eval_slice["queue_priority_reasons"].map(lambda reasons: "uncertain_existing_review" in reasons).sum()
        )
        if not blind_eval_slice.empty
        else 0,
        "high_signal_unreviewed_fallback_candidate_count": int(
            blind_eval_slice["queue_priority_reasons"].map(lambda reasons: "high_signal_unreviewed_fallback" in reasons).sum()
        )
        if not blind_eval_slice.empty
        else 0,
        "unreviewed_candidate_row_count": int(
            blind_eval_slice["current_clinician_reviewed_flag"].eq(0).sum()
        )
        if not blind_eval_slice.empty
        else 0,
        "already_reviewed_candidate_row_count": int(
            blind_eval_slice["current_clinician_reviewed_flag"].eq(1).sum()
        )
        if not blind_eval_slice.empty
        else 0,
        "candidate_rows_by_medication_class": (
            blind_eval_slice["medication_class_standardized"]
            .fillna("unresolved")
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        )
        if not blind_eval_slice.empty
        else {},
        "candidate_distinct_subject_count": int(
            blind_eval_slice["subject_id"].nunique(dropna=True)
        )
        if not blind_eval_slice.empty
        else 0,
        "candidate_distinct_encounter_count": int(
            blind_eval_slice["encounter_id"].nunique(dropna=True)
        )
        if not blind_eval_slice.empty
        else 0,
        "notes": [
            "This slice is generated from the real ordinal logistic model, not from dummy baselines.",
            "model-vs-rule disagreement availability is limited by benchmark__current_rule_score coverage in the current workspace.",
            "The blind reviewer UI hides model predictions, benchmark levels, and queue rationale during the final session to reduce anchoring bias.",
        ],
    }


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
