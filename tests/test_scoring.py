"""Tests for the structured IPD scoring refactor."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from opti_med.scoring.scorer import (
    collect_rule_hits,
    collect_structured_evidence,
    normalize_bucket_scores,
    score_hits_by_bucket,
)


class StructuredScoringTests(unittest.TestCase):
    def test_bucket_scores_are_capped(self) -> None:
        row = pd.Series(
            {
                "drug": "Lorazepam",
                "benzodiazepine_flag": 1,
                "opioid_flag": 1,
                "anticholinergic_flag": 1,
                "antipsychotic_flag": 0,
                "ppi_flag": 0,
                "current_polypharmacy_flag": 1,
                "current_medication_count": 12,
                "historical_medication_count": 15,
                "renal_risk_flag": 1,
                "diagnosis_risk_renal_flag": 1,
                "diagnosis_risk_cognitive_flag": 1,
                "diagnosis_risk_cardiac_flag": 1,
                "diagnosis_risk_metabolic_flag": 0,
                "age_proxy": 88,
                "bmi": 19,
                "egfr_ml_min_1_73m2": 30,
                "creatinine_trend_direction": "rising",
                "creatinine_first": 1.2,
                "creatinine_last": 1.8,
                "creatinine_delta": 0.6,
                "potassium_max": 5.6,
                "sodium_max": 146,
                "sbp_min": 88,
                "heart_rate_max": 120,
                "pain_max": 8,
            }
        )
        hits = collect_rule_hits(row)
        bucket_scores = score_hits_by_bucket(hits)
        self.assertEqual(bucket_scores["base_medication_risk"], 4)
        self.assertEqual(bucket_scores["terrain_aggravating_context"], 3)
        self.assertEqual(bucket_scores["dynamic_biologic_vital_evidence"], 3)
        self.assertEqual(normalize_bucket_scores(bucket_scores), 10)

    def test_structured_evidence_collects_used_values(self) -> None:
        row = pd.Series(
            {
                "drug": "Oxycodone",
                "opioid_flag": 1,
                "benzodiazepine_flag": 0,
                "anticholinergic_flag": 0,
                "antipsychotic_flag": 0,
                "ppi_flag": 0,
                "current_medication_count": 4,
                "current_polypharmacy_flag": 0,
                "renal_risk_flag": 0,
                "diagnosis_risk_renal_flag": 0,
                "diagnosis_risk_cognitive_flag": 0,
                "diagnosis_risk_cardiac_flag": 0,
                "diagnosis_risk_metabolic_flag": 0,
                "creatinine_trend_direction": "stable",
            }
        )
        hits = collect_rule_hits(row)
        evidence = collect_structured_evidence(hits)
        self.assertEqual(evidence["drug"], "Oxycodone")
        self.assertIn("bucket_reasons", evidence)
        self.assertIn("base_medication_risk", evidence["bucket_reasons"])
        json.dumps(evidence)


if __name__ == "__main__":
    unittest.main()
