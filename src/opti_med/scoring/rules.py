"""Transparent scoring rules for the MVP deprescribing priority score."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class ScoringRule:
    """A single transparent scoring rule."""

    name: str
    points: int
    predicate: Callable[[pd.Series], bool]
    explanation: str


SCORING_RULES: tuple[ScoringRule, ...] = (
    ScoringRule(
        name="very_high_medication_burden",
        points=3,
        predicate=lambda row: int(row.get("total_medication_count", 0)) >= 10,
        explanation="very high medication burden (10+ distinct medications)",
    ),
    ScoringRule(
        name="polypharmacy",
        points=2,
        predicate=lambda row: (
            int(row.get("polypharmacy_flag", 0)) == 1
            and int(row.get("total_medication_count", 0)) < 10
        ),
        explanation="polypharmacy during admission",
    ),
    ScoringRule(
        name="renal_risk",
        points=2,
        predicate=lambda row: int(row.get("renal_risk_flag", 0)) == 1,
        explanation="renal risk based on creatinine",
    ),
    ScoringRule(
        name="benzodiazepine",
        points=2,
        predicate=lambda row: int(row.get("benzodiazepine_flag", 0)) == 1,
        explanation="benzodiazepine exposure",
    ),
    ScoringRule(
        name="opioid",
        points=2,
        predicate=lambda row: int(row.get("opioid_flag", 0)) == 1,
        explanation="opioid exposure",
    ),
    ScoringRule(
        name="anticholinergic",
        points=2,
        predicate=lambda row: int(row.get("anticholinergic_flag", 0)) == 1,
        explanation="anticholinergic exposure",
    ),
    ScoringRule(
        name="antipsychotic",
        points=2,
        predicate=lambda row: int(row.get("antipsychotic_flag", 0)) == 1,
        explanation="antipsychotic exposure",
    ),
    ScoringRule(
        name="ppi",
        points=1,
        predicate=lambda row: int(row.get("ppi_flag", 0)) == 1,
        explanation="proton pump inhibitor exposure",
    ),
    ScoringRule(
        name="ckd",
        points=1,
        predicate=lambda row: int(row.get("ckd_flag", 0)) == 1,
        explanation="chronic kidney disease diagnosis",
    ),
    ScoringRule(
        name="dementia",
        points=1,
        predicate=lambda row: int(row.get("dementia_flag", 0)) == 1,
        explanation="dementia diagnosis",
    ),
    ScoringRule(
        name="delirium",
        points=2,
        predicate=lambda row: int(row.get("delirium_flag", 0)) == 1,
        explanation="delirium diagnosis",
    ),
    ScoringRule(
        name="heart_failure",
        points=1,
        predicate=lambda row: int(row.get("heart_failure_flag", 0)) == 1,
        explanation="heart failure diagnosis",
    ),
    ScoringRule(
        name="diabetes",
        points=1,
        predicate=lambda row: int(row.get("diabetes_flag", 0)) == 1,
        explanation="diabetes diagnosis",
    ),
)


def score_to_label(score: int) -> str:
    """Map the numeric score to a simple risk label."""
    if score >= 7:
        return "high"
    if score >= 4:
        return "medium"
    return "low"
