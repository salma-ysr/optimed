"""Tests for patient context helper functions."""

from __future__ import annotations

import unittest

from opti_med.features.context import (
    categorize_delta_trend,
    compute_cockcroft_gault,
    compute_egfr_ckd_epi_2021,
    derive_cockcroft_gault_unavailable_reason,
)


class PatientContextHelperTests(unittest.TestCase):
    def test_compute_egfr_ckd_epi_2021(self) -> None:
        estimate = compute_egfr_ckd_epi_2021(age=70, sex="F", creatinine_mg_dl=1.2)
        self.assertIsNotNone(estimate)
        self.assertGreater(estimate, 0)

    def test_compute_cockcroft_gault_returns_none_without_weight(self) -> None:
        estimate = compute_cockcroft_gault(
            age=70,
            sex="M",
            weight_kg=None,
            creatinine_mg_dl=1.2,
        )
        self.assertIsNone(estimate)

    def test_derive_cockcroft_gault_unavailable_reason(self) -> None:
        reason = derive_cockcroft_gault_unavailable_reason(
            {
                "cockcroft_gault_ml_min": None,
                "weight_kg": None,
                "creatinine_first": 1.0,
                "age_context": 76,
                "sex_context": "F",
            }
        )
        self.assertEqual(reason, "missing_weight")

    def test_categorize_delta_trend(self) -> None:
        self.assertEqual(categorize_delta_trend(0.5), "rising")
        self.assertEqual(categorize_delta_trend(-0.5), "falling")
        self.assertEqual(categorize_delta_trend(0.1), "stable")


if __name__ == "__main__":
    unittest.main()
