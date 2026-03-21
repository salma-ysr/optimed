"""Scoring logic for the MVP deprescribing priority score."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.features.builder import MinimalFeatureBuilder
from opti_med.scoring.rules import (
    BUCKETED_SCORING_RULES,
    SCORE_BUCKETS,
    SCORE_BUCKET_INDEX,
    RuleHit,
    score_to_label,
    score_to_summary_alert,
)


@dataclass(frozen=True)
class ScoreBuildResult:
    """Scored cohort and its output path."""

    dataframe: pd.DataFrame
    output_path: Path
    dropped_duplicate_rows: int = 0


class DeprescribingPriorityScorer:
    """Apply transparent rule-based deprescribing priority scoring."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.feature_builder = MinimalFeatureBuilder(settings)

    def build(self) -> pd.DataFrame:
        """Build the scored dataframe from source tables."""
        processed = self.feature_builder.build()
        return apply_deprescribing_priority_score(processed)

    def save(
        self, dataframe: pd.DataFrame, output_path: Path | None = None
    ) -> ScoreBuildResult:
        """Persist the scored dataframe to a CSV file."""
        target_path = output_path or self.settings.scored_output_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(target_path, index=False)
        return ScoreBuildResult(
            dataframe=dataframe,
            output_path=target_path,
            dropped_duplicate_rows=self.feature_builder.cohort_builder.last_dropped_duplicate_rows,
        )


def apply_deprescribing_priority_score(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add rule-based deprescribing priority outputs to each row."""
    scored = dataframe.copy()
    row_scores = scored.apply(_score_row, axis=1, result_type="expand")
    row_scores.columns = [
        "deprescribing_priority_score",
        "deprescribing_priority_label",
        "deprescribing_priority_summary_alert",
        "deprescribing_priority_explanation",
        "deprescribing_priority_bucket_scores_json",
        "deprescribing_priority_reasons_json",
        "deprescribing_priority_evidence_json",
    ]
    scored = pd.concat([scored, row_scores], axis=1)
    return scored


def summarize_scored_cohort(dataframe: pd.DataFrame) -> list[str]:
    """Return compact summaries for the scored cohort CLI."""
    labels = dataframe["deprescribing_priority_label"].value_counts()
    return [
        f"rows={len(dataframe):,}, columns={dataframe.shape[1]}",
        f"unique_admissions={dataframe['hadm_id'].nunique():,}",
        f"high_priority_rows={int(labels.get('high', 0)):,}",
        f"medium_priority_rows={int(labels.get('medium', 0)):,}",
        f"low_priority_rows={int(labels.get('low', 0)):,}",
    ]


def _score_row(row: pd.Series) -> tuple[int, str, str, str, str, str, str]:
    """Score a single processed cohort row with bucketed structured outputs."""
    hits = collect_rule_hits(row)
    bucket_scores = score_hits_by_bucket(hits)
    final_score = normalize_bucket_scores(bucket_scores)
    label = score_to_label(final_score)
    reasons = [hit.clinician_reason for hit in hits]
    summary_alert = score_to_summary_alert(final_score, hits, bucket_scores)
    explanation = "; ".join(reasons) if reasons else "no major structured IPD rule triggered"
    evidence = collect_structured_evidence(hits)
    return (
        final_score,
        label,
        summary_alert,
        explanation,
        json.dumps(bucket_scores, sort_keys=True),
        json.dumps(reasons),
        json.dumps(evidence, sort_keys=True),
    )


def collect_rule_hits(row: pd.Series) -> list[RuleHit]:
    """Collect all triggered bucketed rule hits for one scored medication row."""
    hits: list[RuleHit] = []
    for rule in BUCKETED_SCORING_RULES:
        if rule.predicate(row):
            hits.append(
                RuleHit(
                    rule_name=rule.name,
                    bucket=rule.bucket,
                    points=rule.points,
                    clinician_reason=rule.clinician_reason,
                    alert_fragment=rule.alert_fragment,
                    evidence=rule.evidence_builder(row),
                )
            )
    return hits


def score_hits_by_bucket(hits: list[RuleHit]) -> dict[str, int]:
    """Cap each score bucket independently so the total remains interpretable."""
    raw_bucket_scores = {bucket.key: 0 for bucket in SCORE_BUCKETS}
    for hit in hits:
        raw_bucket_scores[hit.bucket] += hit.points

    return {
        bucket.key: min(raw_bucket_scores[bucket.key], bucket.max_points)
        for bucket in SCORE_BUCKETS
    }


def normalize_bucket_scores(bucket_scores: dict[str, int]) -> int:
    """Normalize the structured bucket scores to the configured 0-to-10 scale."""
    normalized_score = sum(
        min(bucket_scores.get(bucket.key, 0), bucket.max_points)
        for bucket in SCORE_BUCKETS
    )
    return max(0, min(normalized_score, sum(bucket.max_points for bucket in SCORE_BUCKETS)))


def collect_structured_evidence(hits: list[RuleHit]) -> dict[str, object]:
    """Aggregate the evidence values used by triggered rules into a stable payload."""
    evidence: dict[str, object] = {}
    bucket_reasons: dict[str, list[str]] = {bucket.key: [] for bucket in SCORE_BUCKETS}
    for hit in hits:
        bucket_reasons[hit.bucket].append(hit.clinician_reason)
        for field_name, value in hit.evidence.items():
            if field_name not in evidence:
                evidence[field_name] = value
    evidence["bucket_labels"] = {
        bucket.key: SCORE_BUCKET_INDEX[bucket.key].label
        for bucket in SCORE_BUCKETS
    }
    evidence["bucket_reasons"] = bucket_reasons
    return evidence
