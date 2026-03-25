"""Tests for the structured IPD scoring refactor."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from opti_med.scoring.scorer import (
    apply_deprescribing_priority_score,
    collect_rule_hits,
    collect_structured_evidence,
    normalize_bucket_scores,
    score_hits_by_bucket,
)
from opti_med.scoring.priority_levels import (
    build_priority_level_contract_payload,
    build_priority_comparison,
    coerce_priority_score_value,
    score_to_priority_level,
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

    def test_score_to_priority_level_uses_canonical_buckets(self) -> None:
        self.assertEqual(score_to_priority_level(0), "low")
        self.assertEqual(score_to_priority_level(3.9), "low")
        self.assertEqual(score_to_priority_level(4), "medium")
        self.assertEqual(score_to_priority_level(6.5), "medium")
        self.assertEqual(score_to_priority_level(7), "high")
        self.assertEqual(score_to_priority_level(10), "high")
        self.assertIsNone(score_to_priority_level(None))

    def test_scored_output_carries_level_column(self) -> None:
        dataframe = pd.DataFrame(
            [
                {
                    "drug": "Lorazepam",
                    "benzodiazepine_flag": 1,
                    "opioid_flag": 0,
                    "anticholinergic_flag": 0,
                    "antipsychotic_flag": 0,
                    "ppi_flag": 0,
                    "current_medication_count": 4,
                    "historical_medication_count": 4,
                    "current_polypharmacy_flag": 0,
                    "polypharmacy_flag": 0,
                    "renal_risk_flag": 0,
                    "diagnosis_risk_renal_flag": 0,
                    "diagnosis_risk_cognitive_flag": 0,
                    "diagnosis_risk_cardiac_flag": 0,
                    "diagnosis_risk_metabolic_flag": 0,
                    "age_proxy": 75,
                    "bmi": 25,
                    "egfr_ml_min_1_73m2": 70,
                    "creatinine_trend_direction": "stable",
                    "potassium_max": 4.2,
                    "sodium_max": 140,
                    "sbp_min": 110,
                    "heart_rate_max": 90,
                    "pain_max": 2,
                }
            ]
        )
        scored = apply_deprescribing_priority_score(dataframe)
        self.assertIn("deprescribing_priority_score_level", scored.columns)
        self.assertEqual(
            scored.loc[0, "deprescribing_priority_score_level"],
            scored.loc[0, "deprescribing_priority_label"],
        )

    def test_priority_comparison_prefers_same_level_agreement(self) -> None:
        key_frame = pd.DataFrame({"row_id": ["a", "b", "c"]})
        result = build_priority_comparison(
            key_frame=key_frame,
            left_source_name="left",
            right_source_name="right",
            left_score_series=pd.Series([8, 5, 2]),
            right_score_series=pd.Series([7, 4, 3]),
        )
        self.assertEqual(result.summary["same_level_agreement_count"], 3)
        self.assertEqual(result.summary["same_level_agreement_rate"], 1.0)
        self.assertEqual(result.summary["exact_score_agreement_count"], 0)
        self.assertEqual(
            result.rowwise["comparison__same_level_match_flag"].fillna(False).astype(bool).sum(),
            3,
        )

    def test_priority_comparison_fills_missing_levels_from_scores(self) -> None:
        key_frame = pd.DataFrame({"row_id": ["a", "b"]})
        result = build_priority_comparison(
            key_frame=key_frame,
            left_source_name="left",
            right_source_name="right",
            left_score_series=pd.Series([8, 5]),
            left_level_series=pd.Series([pd.NA, "medium"], dtype="string"),
            right_score_series=pd.Series([7, 4]),
            right_level_series=pd.Series(["high", pd.NA], dtype="string"),
        )
        self.assertEqual(result.summary["same_level_agreement_count"], 2)
        self.assertListEqual(
            result.rowwise["comparison__left_level"].tolist(),
            ["high", "medium"],
        )
        self.assertListEqual(
            result.rowwise["comparison__right_level"].tolist(),
            ["high", "medium"],
        )

    def test_priority_score_contract_rejects_out_of_range_values(self) -> None:
        with self.assertRaises(ValueError):
            coerce_priority_score_value(-0.1)
        with self.assertRaises(ValueError):
            coerce_priority_score_value(10.1)

    def test_priority_level_contract_payload_is_canonical(self) -> None:
        payload = build_priority_level_contract_payload()
        self.assertEqual(payload["mapping"]["low"], "0-3")
        self.assertEqual(payload["mapping"]["medium"], "4-6")
        self.assertEqual(payload["mapping"]["high"], "7-10")
        self.assertEqual(
            payload["float_bucketing_policy"],
            "[0.0, 4.0) => low; [4.0, 7.0) => medium; [7.0, 10.0] => high",
        )


if __name__ == "__main__":
    unittest.main()
