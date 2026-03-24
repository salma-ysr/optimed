"""Schema integrity tests for feature-store contracts using synthetic examples only."""

from __future__ import annotations

import unittest

from opti_med.features.contracts import (
    FEATURE_GROUP_SCHEMAS,
    FEATURE_STORE_GRAIN_DESCRIPTION,
    FEATURE_STORE_PRIMARY_KEY,
    CLASS_BURDEN_COUNTS_SCHEMA,
    LAB_TRAJECTORY_WINDOWS_SCHEMA,
    PRIOR_UTILIZATION_SCHEMA,
    PROVENANCE_MISSINGNESS_FEATURE_GROUP,
    RENAL_VULNERABILITY_SCHEMA,
    VITAL_TRAJECTORY_WINDOWS_SCHEMA,
    ExpectedMissingnessBehavior,
    FeatureDerivationType,
    FeatureFamily,
    FeatureGroupSchema,
    FeatureSpec,
    FRAILTY_SURROGATES_SCHEMA,
    validate_feature_group_schema,
    validate_feature_registry,
)


class FeatureContractsRegistryTests(unittest.TestCase):
    def test_feature_registry_validates(self) -> None:
        validate_feature_registry()

    def test_feature_store_grain_matches_review_time_medication_shape(self) -> None:
        self.assertEqual(
            FEATURE_STORE_GRAIN_DESCRIPTION,
            "one row per subject_id + encounter_id + medication_standardized + review_timestamp",
        )
        self.assertEqual(
            FEATURE_STORE_PRIMARY_KEY,
            ("subject_id", "encounter_id", "medication_standardized", "review_timestamp"),
        )

    def test_placeholder_group_schemas_are_registered(self) -> None:
        for group in [
            CLASS_BURDEN_COUNTS_SCHEMA,
            RENAL_VULNERABILITY_SCHEMA,
            FRAILTY_SURROGATES_SCHEMA,
            PRIOR_UTILIZATION_SCHEMA,
            LAB_TRAJECTORY_WINDOWS_SCHEMA,
            VITAL_TRAJECTORY_WINDOWS_SCHEMA,
        ]:
            self.assertIn(group.group_name, FEATURE_GROUP_SCHEMAS)

    def test_provenance_missingness_family_exists(self) -> None:
        self.assertEqual(
            PROVENANCE_MISSINGNESS_FEATURE_GROUP.family,
            FeatureFamily.PROVENANCE_MISSINGNESS,
        )


class FeatureContractsSyntheticValidationTests(unittest.TestCase):
    def test_synthetic_group_with_valid_metadata_passes(self) -> None:
        synthetic_group = FeatureGroupSchema(
            group_name="synthetic_group",
            family=FeatureFamily.MEDICATION_SEMANTICS,
            purpose="Synthetic test-only group.",
            direct_fields=(
                FeatureSpec(
                    feature_name="synthetic_direct_feature",
                    description="Synthetic direct feature.",
                    family=FeatureFamily.MEDICATION_SEMANTICS,
                    derivation_type=FeatureDerivationType.DIRECT,
                    source_tables=("synthetic_source",),
                    point_in_time_safe=True,
                    expected_missingness_behavior=ExpectedMissingnessBehavior.REQUIRED_IF_STATE_EXISTS,
                    provenance_columns=("synthetic_provenance",),
                ),
            ),
            transformed_fields=(
                FeatureSpec(
                    feature_name="synthetic_transformed_feature",
                    description="Synthetic transformed feature.",
                    family=FeatureFamily.MEDICATION_SEMANTICS,
                    derivation_type=FeatureDerivationType.TRANSFORMED,
                    source_tables=("synthetic_source",),
                    point_in_time_safe=True,
                    expected_missingness_behavior=ExpectedMissingnessBehavior.DERIVABLE_IF_UPSTREAM_PRESENT,
                    provenance_columns=("synthetic_provenance",),
                ),
            ),
            inferred_fields=(
                FeatureSpec(
                    feature_name="synthetic_inferred_feature",
                    description="Synthetic inferred feature.",
                    family=FeatureFamily.MEDICATION_SEMANTICS,
                    derivation_type=FeatureDerivationType.INFERRED,
                    source_tables=("synthetic_source",),
                    point_in_time_safe=True,
                    expected_missingness_behavior=ExpectedMissingnessBehavior.SOURCE_CONDITIONAL,
                    provenance_columns=("synthetic_provenance",),
                ),
            ),
        )
        validate_feature_group_schema(synthetic_group)

    def test_invalid_derivation_placement_raises(self) -> None:
        invalid_group = FeatureGroupSchema(
            group_name="invalid_group",
            family=FeatureFamily.PATIENT_CONTEXT,
            purpose="Synthetic invalid group.",
            direct_fields=(
                FeatureSpec(
                    feature_name="bad_feature",
                    description="Wrong derivation placement.",
                    family=FeatureFamily.PATIENT_CONTEXT,
                    derivation_type=FeatureDerivationType.INFERRED,
                    source_tables=("synthetic_source",),
                    point_in_time_safe=True,
                    expected_missingness_behavior=ExpectedMissingnessBehavior.SOURCE_CONDITIONAL,
                    provenance_columns=("synthetic_provenance",),
                ),
            ),
        )
        with self.assertRaises(ValueError):
            validate_feature_group_schema(invalid_group)

    def test_missing_provenance_columns_raises(self) -> None:
        invalid_group = FeatureGroupSchema(
            group_name="missing_provenance_group",
            family=FeatureFamily.TEMPORAL_PHYSIOLOGY,
            purpose="Synthetic invalid group.",
            direct_fields=(
                FeatureSpec(
                    feature_name="feature_without_provenance",
                    description="Missing provenance columns should fail validation.",
                    family=FeatureFamily.TEMPORAL_PHYSIOLOGY,
                    derivation_type=FeatureDerivationType.DIRECT,
                    source_tables=("synthetic_source",),
                    point_in_time_safe=True,
                    expected_missingness_behavior=ExpectedMissingnessBehavior.WINDOW_CONDITIONAL,
                    provenance_columns=(),
                ),
            ),
        )
        with self.assertRaises(ValueError):
            validate_feature_group_schema(invalid_group)


if __name__ == "__main__":
    unittest.main()
