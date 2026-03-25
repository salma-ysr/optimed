"""Targeted tests for dossier medication class exposure."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from opti_med.api import app as api_app


class DossierMedicationClassExposureTests(unittest.TestCase):
    def test_resolved_classes_use_rxnorm_mapping_fallback_for_supported_salt_forms(self) -> None:
        mapping = pd.DataFrame(
            [
                {
                    "medication_query_key": "fentanyl citrate",
                    "medication_normalized": "fentanyl citrate",
                    "medication_standardized": "fentanyl citrate",
                    "ingredient_standardized": "fentanyl citrate",
                    "class_labels_json": [],
                }
            ]
        )

        with patch.object(api_app, "_load_medication_rxnorm_mapping_dataframe", return_value=mapping):
            classes = api_app._resolved_medication_classes(
                score_record=None,
                review_row={"medication_normalized": "fentanyl citrate"},
                reviewable_row={"medication_class_standardized": "unresolved"},
            )

        self.assertEqual(classes, ["Opioid"])

    def test_resolved_classes_leave_unmapped_medications_empty(self) -> None:
        mapping = pd.DataFrame(
            [
                {
                    "medication_query_key": "acetaminophen iv",
                    "medication_normalized": "acetaminophen iv",
                    "medication_standardized": "acetaminophen",
                    "ingredient_standardized": "acetaminophen",
                    "class_labels_json": [],
                }
            ]
        )

        with patch.object(api_app, "_load_medication_rxnorm_mapping_dataframe", return_value=mapping):
            classes = api_app._resolved_medication_classes(
                score_record=None,
                review_row={"medication_normalized": "acetaminophen iv"},
                reviewable_row=None,
            )

        self.assertEqual(classes, [])
