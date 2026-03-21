"""Configuration-driven structured IPD scoring rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


ScorePredicate = Callable[[pd.Series], bool]
EvidenceBuilder = Callable[[pd.Series], dict[str, object]]


@dataclass(frozen=True)
class ScoreBucket:
    """One top-level score bucket in the structured IPD model."""

    key: str
    label: str
    max_points: int


@dataclass(frozen=True)
class BucketedScoringRule:
    """A single rule contributing to one structured score bucket."""

    name: str
    bucket: str
    points: int
    predicate: ScorePredicate
    clinician_reason: str
    alert_fragment: str
    evidence_builder: EvidenceBuilder


@dataclass(frozen=True)
class RuleHit:
    """A triggered rule with its contribution and evidence."""

    rule_name: str
    bucket: str
    points: int
    clinician_reason: str
    alert_fragment: str
    evidence: dict[str, object]


SCORE_BUCKETS: tuple[ScoreBucket, ...] = (
    ScoreBucket(
        key="base_medication_risk",
        label="Base Medication Risk",
        max_points=4,
    ),
    ScoreBucket(
        key="terrain_aggravating_context",
        label="Terrain / Aggravating Patient Context",
        max_points=3,
    ),
    ScoreBucket(
        key="dynamic_biologic_vital_evidence",
        label="Dynamic Biologic / Vital Evidence",
        max_points=3,
    ),
)

SCORE_BUCKET_INDEX = {bucket.key: bucket for bucket in SCORE_BUCKETS}
SCORE_MAX = sum(bucket.max_points for bucket in SCORE_BUCKETS)


def _evidence(*field_names: str) -> EvidenceBuilder:
    """Build a small evidence payload from the listed row fields."""
    return lambda row: {
        field_name: row.get(field_name)
        for field_name in field_names
        if row.get(field_name) is not None and not pd.isna(row.get(field_name))
    }


BUCKETED_SCORING_RULES: tuple[BucketedScoringRule, ...] = (
    BucketedScoringRule(
        name="benzodiazepine_exposure",
        bucket="base_medication_risk",
        points=3,
        predicate=lambda row: int(row.get("benzodiazepine_flag", 0)) == 1,
        clinician_reason="benzodiazepine exposure",
        alert_fragment="benzodiazepine risk",
        evidence_builder=_evidence("drug", "benzodiazepine_flag"),
    ),
    BucketedScoringRule(
        name="opioid_exposure",
        bucket="base_medication_risk",
        points=3,
        predicate=lambda row: int(row.get("opioid_flag", 0)) == 1,
        clinician_reason="opioid exposure",
        alert_fragment="opioid risk",
        evidence_builder=_evidence("drug", "opioid_flag"),
    ),
    BucketedScoringRule(
        name="anticholinergic_exposure",
        bucket="base_medication_risk",
        points=3,
        predicate=lambda row: int(row.get("anticholinergic_flag", 0)) == 1,
        clinician_reason="anticholinergic exposure",
        alert_fragment="anticholinergic burden",
        evidence_builder=_evidence("drug", "anticholinergic_flag"),
    ),
    BucketedScoringRule(
        name="antipsychotic_exposure",
        bucket="base_medication_risk",
        points=3,
        predicate=lambda row: int(row.get("antipsychotic_flag", 0)) == 1,
        clinician_reason="antipsychotic exposure",
        alert_fragment="antipsychotic risk",
        evidence_builder=_evidence("drug", "antipsychotic_flag"),
    ),
    BucketedScoringRule(
        name="ppi_exposure",
        bucket="base_medication_risk",
        points=1,
        predicate=lambda row: int(row.get("ppi_flag", 0)) == 1,
        clinician_reason="proton pump inhibitor exposure",
        alert_fragment="ppi exposure",
        evidence_builder=_evidence("drug", "ppi_flag"),
    ),
    BucketedScoringRule(
        name="very_high_medication_burden",
        bucket="base_medication_risk",
        points=2,
        predicate=lambda row: int(row.get("total_medication_count", 0)) >= 10,
        clinician_reason="very high medication burden (10+ distinct medications)",
        alert_fragment="very high medication count",
        evidence_builder=_evidence("total_medication_count"),
    ),
    BucketedScoringRule(
        name="polypharmacy",
        bucket="base_medication_risk",
        points=1,
        predicate=lambda row: int(row.get("polypharmacy_flag", 0)) == 1,
        clinician_reason="polypharmacy during admission",
        alert_fragment="polypharmacy",
        evidence_builder=_evidence("total_medication_count", "polypharmacy_flag"),
    ),
    BucketedScoringRule(
        name="renal_terrain",
        bucket="terrain_aggravating_context",
        points=2,
        predicate=lambda row: (
            int(row.get("diagnosis_risk_renal_flag", row.get("ckd_flag", 0))) == 1
            or int(row.get("renal_risk_flag", 0)) == 1
            or (
                pd.to_numeric(pd.Series([row.get("egfr_ml_min_1_73m2")]), errors="coerce").iloc[0] < 45
                if not pd.isna(pd.to_numeric(pd.Series([row.get("egfr_ml_min_1_73m2")]), errors="coerce").iloc[0])
                else False
            )
        ),
        clinician_reason="reduced renal reserve or renal vulnerability",
        alert_fragment="renal vulnerability",
        evidence_builder=_evidence("ckd_flag", "renal_risk_flag", "egfr_ml_min_1_73m2"),
    ),
    BucketedScoringRule(
        name="cognitive_terrain",
        bucket="terrain_aggravating_context",
        points=2,
        predicate=lambda row: int(row.get("diagnosis_risk_cognitive_flag", 0)) == 1,
        clinician_reason="cognitive vulnerability from dementia or delirium context",
        alert_fragment="cognitive vulnerability",
        evidence_builder=_evidence("dementia_flag", "delirium_flag", "diagnosis_risk_cognitive_flag"),
    ),
    BucketedScoringRule(
        name="cardiometabolic_terrain",
        bucket="terrain_aggravating_context",
        points=1,
        predicate=lambda row: (
            int(row.get("diagnosis_risk_cardiac_flag", 0)) == 1
            or int(row.get("diagnosis_risk_metabolic_flag", 0)) == 1
        ),
        clinician_reason="cardiac or metabolic comorbidity may narrow medication tolerance",
        alert_fragment="cardiometabolic terrain",
        evidence_builder=_evidence(
            "heart_failure_flag",
            "diabetes_flag",
            "diagnosis_risk_cardiac_flag",
            "diagnosis_risk_metabolic_flag",
        ),
    ),
    BucketedScoringRule(
        name="frailty_reserve",
        bucket="terrain_aggravating_context",
        points=1,
        predicate=lambda row: (
            pd.to_numeric(pd.Series([row.get("age_proxy")]), errors="coerce").iloc[0] >= 85
            or (
                pd.to_numeric(pd.Series([row.get("bmi")]), errors="coerce").iloc[0] < 20
                if not pd.isna(pd.to_numeric(pd.Series([row.get("bmi")]), errors="coerce").iloc[0])
                else False
            )
        ),
        clinician_reason="advanced age or low reserve may reduce physiologic tolerance",
        alert_fragment="reduced baseline reserve",
        evidence_builder=_evidence("age_proxy", "bmi", "weight_kg"),
    ),
    BucketedScoringRule(
        name="rising_creatinine",
        bucket="dynamic_biologic_vital_evidence",
        points=2,
        predicate=lambda row: str(row.get("creatinine_trend_direction", "")) == "rising",
        clinician_reason="creatinine is rising during the encounter",
        alert_fragment="rising creatinine",
        evidence_builder=_evidence("creatinine_first", "creatinine_last", "creatinine_delta"),
    ),
    BucketedScoringRule(
        name="electrolyte_signal",
        bucket="dynamic_biologic_vital_evidence",
        points=1,
        predicate=lambda row: (
            _outside_range(row.get("potassium_max"), lower=3.5, upper=5.2)
            or _outside_range(row.get("sodium_max"), lower=135, upper=145)
        ),
        clinician_reason="electrolyte abnormality may increase medication-related instability",
        alert_fragment="electrolyte concern",
        evidence_builder=_evidence("potassium_min", "potassium_max", "sodium_min", "sodium_max"),
    ),
    BucketedScoringRule(
        name="hemodynamic_signal",
        bucket="dynamic_biologic_vital_evidence",
        points=1,
        predicate=lambda row: (
            _outside_range(row.get("sbp_min"), lower=100, upper=None)
            or _outside_range(row.get("heart_rate_max"), lower=None, upper=110, invert_upper=True)
        ),
        clinician_reason="blood pressure or heart-rate instability is present",
        alert_fragment="hemodynamic instability",
        evidence_builder=_evidence("sbp_min", "sbp_max", "heart_rate_min", "heart_rate_max"),
    ),
    BucketedScoringRule(
        name="pain_signal",
        bucket="dynamic_biologic_vital_evidence",
        points=1,
        predicate=lambda row: (
            pd.to_numeric(pd.Series([row.get("pain_max")]), errors="coerce").iloc[0] >= 7
            if not pd.isna(pd.to_numeric(pd.Series([row.get("pain_max")]), errors="coerce").iloc[0])
            else False
        ),
        clinician_reason="high pain burden may complicate deprescribing prioritization",
        alert_fragment="high pain burden",
        evidence_builder=_evidence("pain_min", "pain_max", "pain_mean"),
    ),
)


def _outside_range(
    value: object,
    *,
    lower: float | None,
    upper: float | None,
    invert_upper: bool = False,
) -> bool:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return False
    if lower is not None and numeric_value < lower:
        return True
    if upper is not None:
        return numeric_value > upper if not invert_upper else numeric_value >= upper
    return False


def score_to_label(score: int) -> str:
    """Map the normalized 0-to-10 score to a simple severity label."""
    if score >= 7:
        return "high"
    if score >= 4:
        return "medium"
    return "low"


def score_to_summary_alert(score: int, hits: list[RuleHit], bucket_scores: dict[str, int]) -> str:
    """Build a short clinician-readable summary alert."""
    if score == 0 or not hits:
        return "No strong immediate deprescribing alert from the current structured evidence."

    dominant_bucket = max(bucket_scores, key=lambda bucket_key: bucket_scores[bucket_key])
    dominant_hits = [hit for hit in hits if hit.bucket == dominant_bucket]
    alert_fragments = [hit.alert_fragment for hit in dominant_hits[:2]]
    summary_stem = {
        "base_medication_risk": "Medication risk signal",
        "terrain_aggravating_context": "Patient context aggravates medication risk",
        "dynamic_biologic_vital_evidence": "Dynamic clinical evidence raises concern",
    }.get(dominant_bucket, "Structured IPD signal")
    return f"{summary_stem}: {', '.join(alert_fragments)}."
