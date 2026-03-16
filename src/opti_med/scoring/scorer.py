"""Scoring logic for the MVP deprescribing priority score."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.features.builder import MinimalFeatureBuilder
from opti_med.scoring.rules import SCORING_RULES, score_to_label


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
        "deprescribing_priority_explanation",
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


def _score_row(row: pd.Series) -> tuple[int, str, str]:
    """Score a single processed cohort row."""
    total_points = 0
    explanations: list[str] = []
    for rule in SCORING_RULES:
        if rule.predicate(row):
            total_points += rule.points
            explanations.append(rule.explanation)

    bounded_score = max(1, min(total_points, 10))
    label = score_to_label(bounded_score)
    explanation = "; ".join(explanations) if explanations else "no major rule triggered"
    return bounded_score, label, explanation
