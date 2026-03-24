"""Synthetic contract tests for label strategy enums, constants, and dataclasses."""

from __future__ import annotations

import unittest
from datetime import datetime

from opti_med.labels.contracts import (
    ANALYTICAL_LABEL_GRAIN_DESCRIPTION,
    ANALYTICAL_LABEL_PRIMARY_KEY,
    AUXILIARY_HARM_LABEL_NAME,
    EXCLUSION_FLAG_NAME,
    PRIMARY_ACTION_LABEL_NAME,
    PROHIBITED_TRAINING_LABELS,
    AuxiliaryHarmLabelCategory,
    AuxiliaryHarmLabelsRow,
    ExclusionCategory,
    ExclusionFlagsRow,
    LabelProvenanceMetadata,
    LabelSourceCategory,
    LabelWindowMetadata,
    PrimaryActionLabelCategory,
    PrimaryActionLabelRow,
)


class LabelContractConstantsTests(unittest.TestCase):
    def test_analytical_label_grain_matches_review_time_medication_shape(self) -> None:
        self.assertEqual(
            ANALYTICAL_LABEL_GRAIN_DESCRIPTION,
            "one row per subject_id + encounter_id + medication_standardized + review_timestamp",
        )
        self.assertEqual(
            ANALYTICAL_LABEL_PRIMARY_KEY,
            ("subject_id", "encounter_id", "medication_standardized", "review_timestamp"),
        )

    def test_prohibited_training_labels_include_current_rule_score_outputs(self) -> None:
        self.assertIn("deprescribing_priority_score", PROHIBITED_TRAINING_LABELS)
        self.assertIn("deprescribing_priority_label", PROHIBITED_TRAINING_LABELS)
        self.assertIn("current_rule_score", PROHIBITED_TRAINING_LABELS)

    def test_named_label_contracts_are_stable(self) -> None:
        self.assertEqual(PRIMARY_ACTION_LABEL_NAME, "primary_action_label")
        self.assertEqual(AUXILIARY_HARM_LABEL_NAME, "auxiliary_harm_labels")
        self.assertEqual(EXCLUSION_FLAG_NAME, "exclusion_flags")


class LabelContractEnumTests(unittest.TestCase):
    def test_primary_action_label_categories_are_explicit(self) -> None:
        self.assertEqual(
            PrimaryActionLabelCategory.STOPPED_OR_DEINTENSIFIED_BEFORE_DISCHARGE.value,
            "stopped_or_deintensified_before_discharge",
        )

    def test_auxiliary_harm_categories_cover_requested_targets(self) -> None:
        self.assertEqual(
            tuple(category.value for category in AuxiliaryHarmLabelCategory),
            (
                "renal_deterioration",
                "hemodynamic_instability",
                "electrolyte_instability",
                "oversedation_respiratory_risk",
            ),
        )

    def test_exclusion_categories_cover_requested_cases(self) -> None:
        self.assertEqual(
            tuple(category.value for category in ExclusionCategory),
            (
                "acute_life_sustaining_medication",
                "fundamentally_different_deprescribing_logic",
            ),
        )


class LabelContractDataclassTests(unittest.TestCase):
    def test_primary_action_label_row_instantiates_with_synthetic_metadata(self) -> None:
        window = LabelWindowMetadata(
            review_timestamp=datetime.fromisoformat("2125-03-20 09:00:00"),
            action_window_start=datetime.fromisoformat("2125-03-20 09:00:00"),
            action_window_end=datetime.fromisoformat("2125-03-20 18:00:00"),
            harm_window_start=datetime.fromisoformat("2125-03-20 09:00:00"),
            harm_window_end=datetime.fromisoformat("2125-03-20 18:00:00"),
            discharge_timestamp=datetime.fromisoformat("2125-03-20 20:00:00"),
        )
        provenance = LabelProvenanceMetadata(
            label_definition_version="0.1",
            source_categories=(
                LabelSourceCategory.PRESCRIPTION_CHANGE_EVIDENCE,
                LabelSourceCategory.CONSTRUCTED_LABEL_LOGIC,
            ),
            source_tables=("prescriptions", "encounter_medication_state"),
        )
        row = PrimaryActionLabelRow(
            subject_id=1,
            encounter_id="hadm:10",
            review_timestamp=datetime.fromisoformat("2125-03-20 09:00:00"),
            medication_standardized=None,
            primary_action_label=PrimaryActionLabelCategory.ACTION_UNDETERMINED,
            window=window,
            provenance=provenance,
        )
        self.assertEqual(row.primary_action_label, PrimaryActionLabelCategory.ACTION_UNDETERMINED)
        self.assertTrue(row.provenance.constructed_label_flag)
        self.assertFalse(row.provenance.rule_score_used_as_training_label)

    def test_auxiliary_harm_rows_expose_supported_categories(self) -> None:
        self.assertEqual(
            tuple(category.value for category in AuxiliaryHarmLabelsRow.supported_categories()),
            (
                "renal_deterioration",
                "hemodynamic_instability",
                "electrolyte_instability",
                "oversedation_respiratory_risk",
            ),
        )

    def test_exclusion_rows_expose_supported_categories(self) -> None:
        self.assertEqual(
            tuple(category.value for category in ExclusionFlagsRow.supported_categories()),
            (
                "acute_life_sustaining_medication",
                "fundamentally_different_deprescribing_logic",
            ),
        )


if __name__ == "__main__":
    unittest.main()
