"""Feature-store contracts and schema governance for future ML features.

This module defines the shape and governance metadata for future point-in-time
features without calculating them yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from opti_med.contracts.pipeline import (
    ANALYTICAL_GRAIN_DESCRIPTION,
    ANALYTICAL_GRAIN_PRIMARY_KEY,
)


FEATURE_STORE_GRAIN_DESCRIPTION = ANALYTICAL_GRAIN_DESCRIPTION
FEATURE_STORE_PRIMARY_KEY = ANALYTICAL_GRAIN_PRIMARY_KEY

FEATURE_STORE_DESIGN_PRINCIPLES: tuple[str, ...] = (
    "Preserve provenance richness instead of flattening it away during the ML pivot.",
    "Keep missingness explicit and auditable rather than silently imputing it.",
    "Require point-in-time safety for all registered feature-store features.",
    "Separate direct fields, transformed fields, and inferred fields for every feature group.",
    "Treat feature schema metadata as a contract, not an afterthought.",
)


class FeatureFamily(str, Enum):
    """Top-level typed families for future feature-store content."""

    MEDICATION_SEMANTICS = "medication_semantics_features"
    MEDICATION_BURDEN = "medication_burden_features"
    PATIENT_CONTEXT = "patient_context_features"
    TEMPORAL_PHYSIOLOGY = "temporal_physiology_features"
    PROVENANCE_MISSINGNESS = "provenance_missingness_features"


class FeatureDerivationType(str, Enum):
    """How a feature value is produced."""

    DIRECT = "direct"
    TRANSFORMED = "transformed"
    INFERRED = "inferred"


class ExpectedMissingnessBehavior(str, Enum):
    """Expected missingness behavior for a future feature."""

    REQUIRED_IF_STATE_EXISTS = "required_if_state_exists"
    SOURCE_CONDITIONAL = "source_conditional"
    DERIVABLE_IF_UPSTREAM_PRESENT = "derivable_if_upstream_present"
    WINDOW_CONDITIONAL = "window_conditional"
    STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED = "structural_placeholder_until_implemented"
    MUST_BE_EXPLICITLY_TRACKED = "must_be_explicitly_tracked"


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One governed feature definition."""

    feature_name: str
    description: str
    family: FeatureFamily
    derivation_type: FeatureDerivationType
    source_tables: tuple[str, ...]
    point_in_time_safe: bool
    expected_missingness_behavior: ExpectedMissingnessBehavior
    provenance_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FeatureGroupSchema:
    """One typed feature group with explicit direct/transformed/inferred sections."""

    group_name: str
    family: FeatureFamily
    purpose: str
    direct_fields: tuple[FeatureSpec, ...] = ()
    transformed_fields: tuple[FeatureSpec, ...] = ()
    inferred_fields: tuple[FeatureSpec, ...] = ()

    def all_fields(self) -> tuple[FeatureSpec, ...]:
        """Return all registered features in stable direct/transformed/inferred order."""
        return (
            *self.direct_fields,
            *self.transformed_fields,
            *self.inferred_fields,
        )


DEFAULT_STATE_PROVENANCE_COLUMNS = (
    "review_timestamp_source",
    "source_home_medrecon_flag",
    "source_ed_pyxis_flag",
    "source_hospital_order_flag",
    "source_hospital_admin_flag",
)

DEFAULT_FEATURE_PROVENANCE_COLUMNS = (
    "feature_available_as_of_review_time_flag",
    "review_time_validated_flag",
    "feature_window_end_at_or_before_review_time_flag",
    "source_tables_json",
)

CURRENT_CONTEXT_PROVENANCE_COLUMNS = (
    "diagnosis_context_provenance",
    "ed_diagnosis_context_provenance",
    "age_provenance",
    "sex_provenance",
    "weight_kg_provenance",
    "weight_kg_unavailable_reason",
    "bmi_provenance",
    "bmi_unavailable_reason",
    "creatinine_provenance",
    "creatinine_missingness",
    "potassium_provenance",
    "potassium_missingness",
    "sodium_provenance",
    "sodium_missingness",
    "egfr_provenance",
    "egfr_unavailable_reason",
    "cockcroft_gault_provenance",
    "cockcroft_gault_unavailable_reason",
    "blood_pressure_provenance",
    "heart_rate_provenance",
    "pain_provenance",
    "ed_triage_provenance",
)


MEDICATION_SEMANTICS_FEATURE_GROUP = FeatureGroupSchema(
    group_name="medication_semantics_features",
    family=FeatureFamily.MEDICATION_SEMANTICS,
    purpose=(
        "Feature family covering normalized text, standardized identities, class semantics, schedule semantics, and dose normalization."
    ),
    direct_fields=(
        FeatureSpec(
            feature_name="medication_normalized",
            description="Normalized medication text carried from encounter-medication-state.",
            family=FeatureFamily.MEDICATION_SEMANTICS,
            derivation_type=FeatureDerivationType.DIRECT,
            source_tables=("encounter_medication_state",),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.REQUIRED_IF_STATE_EXISTS,
            provenance_columns=(
                *DEFAULT_STATE_PROVENANCE_COLUMNS,
                "semantic_mapper_name",
                "semantic_mapper_version",
            ),
        ),
        FeatureSpec(
            feature_name="route",
            description="Structured route text available at review time when present in source evidence.",
            family=FeatureFamily.MEDICATION_SEMANTICS,
            derivation_type=FeatureDerivationType.DIRECT,
            source_tables=("encounter_medication_state",),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.SOURCE_CONDITIONAL,
            provenance_columns=DEFAULT_STATE_PROVENANCE_COLUMNS,
        ),
        FeatureSpec(
            feature_name="frequency",
            description="Structured frequency text available at review time when present in source evidence.",
            family=FeatureFamily.MEDICATION_SEMANTICS,
            derivation_type=FeatureDerivationType.DIRECT,
            source_tables=("encounter_medication_state",),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.SOURCE_CONDITIONAL,
            provenance_columns=DEFAULT_STATE_PROVENANCE_COLUMNS,
        ),
    ),
    transformed_fields=(
        FeatureSpec(
            feature_name="dose_normalized",
            description="Future normalized dose representation for burden and mismatch features.",
            family=FeatureFamily.MEDICATION_SEMANTICS,
            derivation_type=FeatureDerivationType.TRANSFORMED,
            source_tables=("encounter_medication_state", "medication_semantics_mapper"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=(
                *DEFAULT_STATE_PROVENANCE_COLUMNS,
                "dose_normalization_status",
                "semantic_mapper_name",
                "semantic_mapper_version",
            ),
        ),
    ),
    inferred_fields=(
        FeatureSpec(
            feature_name="medication_standardized",
            description="Future standardized ingredient identity distinct from normalized text.",
            family=FeatureFamily.MEDICATION_SEMANTICS,
            derivation_type=FeatureDerivationType.INFERRED,
            source_tables=("encounter_medication_state", "medication_semantics_mapper"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=(
                *DEFAULT_STATE_PROVENANCE_COLUMNS,
                "standardization_status",
                "identity_system",
                "identity_system_version",
                "semantic_mapper_name",
                "semantic_mapper_version",
            ),
        ),
        FeatureSpec(
            feature_name="medication_class_standardized",
            description="Future ontology-backed class identity distinct from ingredient standardization and text normalization.",
            family=FeatureFamily.MEDICATION_SEMANTICS,
            derivation_type=FeatureDerivationType.INFERRED,
            source_tables=("encounter_medication_state", "medication_semantics_mapper"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=(
                *DEFAULT_STATE_PROVENANCE_COLUMNS,
                "class_system",
                "class_system_version",
                "semantic_mapper_name",
                "semantic_mapper_version",
            ),
        ),
        FeatureSpec(
            feature_name="scheduled_vs_prn",
            description="Future schedule semantics inferred from medication context.",
            family=FeatureFamily.MEDICATION_SEMANTICS,
            derivation_type=FeatureDerivationType.INFERRED,
            source_tables=("encounter_medication_state", "medication_semantics_mapper"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=(
                *DEFAULT_STATE_PROVENANCE_COLUMNS,
                "schedule_semantics_status",
                "semantic_mapper_name",
                "semantic_mapper_version",
            ),
        ),
    ),
)


MEDICATION_BURDEN_FEATURE_GROUP = FeatureGroupSchema(
    group_name="medication_burden_features",
    family=FeatureFamily.MEDICATION_BURDEN,
    purpose=(
        "Feature family covering counts, concurrency, class burden placeholders, and burden-derived risk views."
    ),
    direct_fields=(
        FeatureSpec(
            feature_name="current_active_medication_count",
            description="Count of active medications at review time from encounter-medication-state rows.",
            family=FeatureFamily.MEDICATION_BURDEN,
            derivation_type=FeatureDerivationType.DIRECT,
            source_tables=("encounter_medication_state",),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.DERIVABLE_IF_UPSTREAM_PRESENT,
            provenance_columns=(
                *DEFAULT_STATE_PROVENANCE_COLUMNS,
                *DEFAULT_FEATURE_PROVENANCE_COLUMNS,
            ),
        ),
        FeatureSpec(
            feature_name="historical_medication_count",
            description="Count of review-time medication history rows not currently active at review time.",
            family=FeatureFamily.MEDICATION_BURDEN,
            derivation_type=FeatureDerivationType.DIRECT,
            source_tables=("encounter_medication_state",),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.DERIVABLE_IF_UPSTREAM_PRESENT,
            provenance_columns=(
                *DEFAULT_STATE_PROVENANCE_COLUMNS,
                *DEFAULT_FEATURE_PROVENANCE_COLUMNS,
            ),
        ),
    ),
    transformed_fields=(
        FeatureSpec(
            feature_name="current_peak_concurrent_medication_count",
            description="Peak concurrent medication burden inside the time-safe review window.",
            family=FeatureFamily.MEDICATION_BURDEN,
            derivation_type=FeatureDerivationType.TRANSFORMED,
            source_tables=("encounter_medication_state",),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.DERIVABLE_IF_UPSTREAM_PRESENT,
            provenance_columns=(
                *DEFAULT_FEATURE_PROVENANCE_COLUMNS,
                "medication_burden_window_start",
                "medication_burden_window_end",
            ),
        ),
        FeatureSpec(
            feature_name="current_polypharmacy_flag",
            description="Thresholded current medication burden flag calculated from active medication count.",
            family=FeatureFamily.MEDICATION_BURDEN,
            derivation_type=FeatureDerivationType.TRANSFORMED,
            source_tables=("encounter_medication_state",),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.DERIVABLE_IF_UPSTREAM_PRESENT,
            provenance_columns=(
                *DEFAULT_FEATURE_PROVENANCE_COLUMNS,
                "polypharmacy_threshold_version",
            ),
        ),
    ),
    inferred_fields=(
        FeatureSpec(
            feature_name="duplicate_therapy_placeholder_flag",
            description="Future duplicate-therapy burden placeholder tied to standardized medication semantics.",
            family=FeatureFamily.MEDICATION_BURDEN,
            derivation_type=FeatureDerivationType.INFERRED,
            source_tables=("encounter_medication_state", "medication_semantics_mapper"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=(
                *DEFAULT_FEATURE_PROVENANCE_COLUMNS,
                "duplicate_therapy_definition_version",
            ),
        ),
    ),
)


PATIENT_CONTEXT_FEATURE_GROUP = FeatureGroupSchema(
    group_name="patient_context_features",
    family=FeatureFamily.PATIENT_CONTEXT,
    purpose=(
        "Feature family covering demographics, diagnoses, reserve proxies, and clinically interpretable context."
    ),
    direct_fields=(
        FeatureSpec(
            feature_name="age_context",
            description="Patient age context available at review time from patient-level source data.",
            family=FeatureFamily.PATIENT_CONTEXT,
            derivation_type=FeatureDerivationType.DIRECT,
            source_tables=("patients", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.REQUIRED_IF_STATE_EXISTS,
            provenance_columns=(
                "age_provenance",
                *DEFAULT_FEATURE_PROVENANCE_COLUMNS,
            ),
        ),
        FeatureSpec(
            feature_name="sex_context",
            description="Patient sex context available at review time from patient-level source data.",
            family=FeatureFamily.PATIENT_CONTEXT,
            derivation_type=FeatureDerivationType.DIRECT,
            source_tables=("patients", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.REQUIRED_IF_STATE_EXISTS,
            provenance_columns=(
                "sex_provenance",
                *DEFAULT_FEATURE_PROVENANCE_COLUMNS,
            ),
        ),
        FeatureSpec(
            feature_name="diagnosis_risk_flags",
            description="Diagnosis-derived risk categories linked to the encounter before review time.",
            family=FeatureFamily.PATIENT_CONTEXT,
            derivation_type=FeatureDerivationType.DIRECT,
            source_tables=("diagnoses_icd", "ed_diagnosis", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.SOURCE_CONDITIONAL,
            provenance_columns=(
                "diagnosis_context_provenance",
                "ed_diagnosis_context_provenance",
                *DEFAULT_FEATURE_PROVENANCE_COLUMNS,
            ),
        ),
    ),
    transformed_fields=(
        FeatureSpec(
            feature_name="egfr_ml_min_1_73m2",
            description="Renal reserve surrogate derived from review-time-safe creatinine and patient context.",
            family=FeatureFamily.PATIENT_CONTEXT,
            derivation_type=FeatureDerivationType.TRANSFORMED,
            source_tables=("labevents", "patients", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.DERIVABLE_IF_UPSTREAM_PRESENT,
            provenance_columns=(
                "egfr_provenance",
                "egfr_unavailable_reason",
                *DEFAULT_FEATURE_PROVENANCE_COLUMNS,
            ),
        ),
        FeatureSpec(
            feature_name="cockcroft_gault_ml_min",
            description="Dose-relevant renal reserve surrogate derived from age, sex, weight, and creatinine when available.",
            family=FeatureFamily.PATIENT_CONTEXT,
            derivation_type=FeatureDerivationType.TRANSFORMED,
            source_tables=("labevents", "patients", "omr", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.SOURCE_CONDITIONAL,
            provenance_columns=(
                "cockcroft_gault_provenance",
                "cockcroft_gault_unavailable_reason",
                *DEFAULT_FEATURE_PROVENANCE_COLUMNS,
            ),
        ),
    ),
    inferred_fields=(
        FeatureSpec(
            feature_name="renal_vulnerability",
            description="Future higher-order renal vulnerability feature group derived from labs, diagnoses, and medication semantics.",
            family=FeatureFamily.PATIENT_CONTEXT,
            derivation_type=FeatureDerivationType.INFERRED,
            source_tables=("diagnoses_icd", "labevents", "encounter_medication_state", "medication_semantics_mapper"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=(
                "renal_vulnerability_definition_version",
                *DEFAULT_FEATURE_PROVENANCE_COLUMNS,
            ),
        ),
        FeatureSpec(
            feature_name="frailty_surrogates",
            description="Future frailty surrogate feature group derived from reserve, morphology, and utilization context.",
            family=FeatureFamily.PATIENT_CONTEXT,
            derivation_type=FeatureDerivationType.INFERRED,
            source_tables=("patients", "omr", "admissions", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=(
                "frailty_surrogates_definition_version",
                *DEFAULT_FEATURE_PROVENANCE_COLUMNS,
            ),
        ),
        FeatureSpec(
            feature_name="prior_utilization",
            description="Future pre-review utilization summary constrained to evidence available at or before review time.",
            family=FeatureFamily.PATIENT_CONTEXT,
            derivation_type=FeatureDerivationType.INFERRED,
            source_tables=("admissions", "edstays", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=(
                "prior_utilization_window_definition",
                *DEFAULT_FEATURE_PROVENANCE_COLUMNS,
            ),
        ),
    ),
)


TEMPORAL_PHYSIOLOGY_FEATURE_GROUP = FeatureGroupSchema(
    group_name="temporal_physiology_features",
    family=FeatureFamily.TEMPORAL_PHYSIOLOGY,
    purpose=(
        "Feature family covering lab and vital trajectories computed inside review-time-safe windows."
    ),
    direct_fields=(
        FeatureSpec(
            feature_name="creatinine_first_last",
            description="Boundary creatinine values observed within the feature window.",
            family=FeatureFamily.TEMPORAL_PHYSIOLOGY,
            derivation_type=FeatureDerivationType.DIRECT,
            source_tables=("labevents", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.WINDOW_CONDITIONAL,
            provenance_columns=(
                "creatinine_provenance",
                "creatinine_missingness",
                *DEFAULT_FEATURE_PROVENANCE_COLUMNS,
            ),
        ),
        FeatureSpec(
            feature_name="vital_extrema",
            description="Review-time-safe extrema for blood pressure, heart rate, and pain signals.",
            family=FeatureFamily.TEMPORAL_PHYSIOLOGY,
            derivation_type=FeatureDerivationType.DIRECT,
            source_tables=("triage", "vitalsign", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.WINDOW_CONDITIONAL,
            provenance_columns=(
                "blood_pressure_provenance",
                "heart_rate_provenance",
                "pain_provenance",
                *DEFAULT_FEATURE_PROVENANCE_COLUMNS,
            ),
        ),
    ),
    transformed_fields=(
        FeatureSpec(
            feature_name="lab_trajectory_windows",
            description="Future lab-trajectory feature group with explicitly governed observation windows.",
            family=FeatureFamily.TEMPORAL_PHYSIOLOGY,
            derivation_type=FeatureDerivationType.TRANSFORMED,
            source_tables=("labevents", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=(
                "lab_window_start",
                "lab_window_end",
                "feature_window_end_at_or_before_review_time_flag",
                *DEFAULT_FEATURE_PROVENANCE_COLUMNS,
            ),
        ),
        FeatureSpec(
            feature_name="vital_trajectory_windows",
            description="Future vital-trajectory feature group with explicitly governed observation windows.",
            family=FeatureFamily.TEMPORAL_PHYSIOLOGY,
            derivation_type=FeatureDerivationType.TRANSFORMED,
            source_tables=("triage", "vitalsign", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=(
                "vital_window_start",
                "vital_window_end",
                "feature_window_end_at_or_before_review_time_flag",
                *DEFAULT_FEATURE_PROVENANCE_COLUMNS,
            ),
        ),
    ),
    inferred_fields=(
        FeatureSpec(
            feature_name="physiology_instability_flags",
            description="Future inferred instability flags derived from safe temporal physiology windows.",
            family=FeatureFamily.TEMPORAL_PHYSIOLOGY,
            derivation_type=FeatureDerivationType.INFERRED,
            source_tables=("labevents", "triage", "vitalsign", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.DERIVABLE_IF_UPSTREAM_PRESENT,
            provenance_columns=DEFAULT_FEATURE_PROVENANCE_COLUMNS,
        ),
    ),
)


PROVENANCE_MISSINGNESS_FEATURE_GROUP = FeatureGroupSchema(
    group_name="provenance_missingness_features",
    family=FeatureFamily.PROVENANCE_MISSINGNESS,
    purpose=(
        "Feature family that keeps source lineage, missingness, and point-in-time safety first-class instead of implicit."
    ),
    direct_fields=(
        FeatureSpec(
            feature_name="feature_available_as_of_review_time_flag",
            description="Explicit flag indicating whether the feature was available by review time.",
            family=FeatureFamily.PROVENANCE_MISSINGNESS,
            derivation_type=FeatureDerivationType.DIRECT,
            source_tables=("encounter_medication_state",),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.MUST_BE_EXPLICITLY_TRACKED,
            provenance_columns=("review_timestamp_source", "review_time_validated_flag"),
        ),
        FeatureSpec(
            feature_name="review_time_validated_flag",
            description="Explicit flag indicating review-time validation passed for the feature row.",
            family=FeatureFamily.PROVENANCE_MISSINGNESS,
            derivation_type=FeatureDerivationType.DIRECT,
            source_tables=("encounter_medication_state", "time_semantics_policy"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.MUST_BE_EXPLICITLY_TRACKED,
            provenance_columns=("review_timestamp_source", "review_time_policy_name"),
        ),
    ),
    transformed_fields=(
        FeatureSpec(
            feature_name="source_tables_json",
            description="Canonical serialized source-table lineage for the feature value.",
            family=FeatureFamily.PROVENANCE_MISSINGNESS,
            derivation_type=FeatureDerivationType.TRANSFORMED,
            source_tables=("encounter_medication_state",),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.MUST_BE_EXPLICITLY_TRACKED,
            provenance_columns=("source_tables_json",),
        ),
    ),
    inferred_fields=(
        FeatureSpec(
            feature_name="missingness_profile",
            description="Explicit machine-readable profile of whether missingness is structural, conditional, or unexpected.",
            family=FeatureFamily.PROVENANCE_MISSINGNESS,
            derivation_type=FeatureDerivationType.INFERRED,
            source_tables=("encounter_medication_state",),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.MUST_BE_EXPLICITLY_TRACKED,
            provenance_columns=(
                *CURRENT_CONTEXT_PROVENANCE_COLUMNS,
                *DEFAULT_FEATURE_PROVENANCE_COLUMNS,
            ),
        ),
    ),
)


CLASS_BURDEN_COUNTS_SCHEMA = FeatureGroupSchema(
    group_name="class_burden_counts",
    family=FeatureFamily.MEDICATION_BURDEN,
    purpose="Placeholder schema for burden features keyed by standardized medication class membership.",
    transformed_fields=(
        FeatureSpec(
            feature_name="opioid_class_count",
            description="Count of opioid-class medications at review time once class identities are standardized.",
            family=FeatureFamily.MEDICATION_BURDEN,
            derivation_type=FeatureDerivationType.TRANSFORMED,
            source_tables=("encounter_medication_state", "medication_semantics_mapper"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=("class_system", "class_system_version", *DEFAULT_FEATURE_PROVENANCE_COLUMNS),
        ),
        FeatureSpec(
            feature_name="sedative_burden",
            description="Future sedative burden aggregate derived from standardized class semantics and dose context.",
            family=FeatureFamily.MEDICATION_BURDEN,
            derivation_type=FeatureDerivationType.TRANSFORMED,
            source_tables=("encounter_medication_state", "medication_semantics_mapper"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=("class_system", "semantic_mapper_name", *DEFAULT_FEATURE_PROVENANCE_COLUMNS),
        ),
    ),
    inferred_fields=(
        FeatureSpec(
            feature_name="opioid_mme",
            description="Future opioid morphine-equivalent burden placeholder.",
            family=FeatureFamily.MEDICATION_BURDEN,
            derivation_type=FeatureDerivationType.INFERRED,
            source_tables=("encounter_medication_state", "medication_semantics_mapper"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=("dose_normalization_status", "semantic_mapper_name", *DEFAULT_FEATURE_PROVENANCE_COLUMNS),
        ),
        FeatureSpec(
            feature_name="anticholinergic_burden",
            description="Future anticholinergic burden placeholder.",
            family=FeatureFamily.MEDICATION_BURDEN,
            derivation_type=FeatureDerivationType.INFERRED,
            source_tables=("encounter_medication_state", "medication_semantics_mapper"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=("class_system", "semantic_mapper_name", *DEFAULT_FEATURE_PROVENANCE_COLUMNS),
        ),
    ),
)


RENAL_VULNERABILITY_SCHEMA = FeatureGroupSchema(
    group_name="renal_vulnerability",
    family=FeatureFamily.PATIENT_CONTEXT,
    purpose="Placeholder schema for renal vulnerability features combining reserve, trajectory, and medication semantics.",
    direct_fields=(
        FeatureSpec(
            feature_name="renal_reserve_inputs_available_flag",
            description="Indicates whether core renal reserve inputs are available by review time.",
            family=FeatureFamily.PATIENT_CONTEXT,
            derivation_type=FeatureDerivationType.DIRECT,
            source_tables=("labevents", "patients", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.MUST_BE_EXPLICITLY_TRACKED,
            provenance_columns=("creatinine_provenance", "egfr_provenance", *DEFAULT_FEATURE_PROVENANCE_COLUMNS),
        ),
    ),
    inferred_fields=(
        FeatureSpec(
            feature_name="renal_dose_mismatch",
            description="Future renal dose mismatch placeholder requiring semantics plus renal reserve features.",
            family=FeatureFamily.PATIENT_CONTEXT,
            derivation_type=FeatureDerivationType.INFERRED,
            source_tables=("labevents", "encounter_medication_state", "medication_semantics_mapper"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=("dose_normalization_status", "egfr_provenance", *DEFAULT_FEATURE_PROVENANCE_COLUMNS),
        ),
    ),
)


FRAILTY_SURROGATES_SCHEMA = FeatureGroupSchema(
    group_name="frailty_surrogates",
    family=FeatureFamily.PATIENT_CONTEXT,
    purpose="Placeholder schema for frailty-adjacent proxies derived from reserve, age, morphology, and utilization context.",
    direct_fields=(
        FeatureSpec(
            feature_name="baseline_reserve_inputs_available_flag",
            description="Indicates whether reserve-related inputs are available for frailty proxy construction.",
            family=FeatureFamily.PATIENT_CONTEXT,
            derivation_type=FeatureDerivationType.DIRECT,
            source_tables=("patients", "omr", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.MUST_BE_EXPLICITLY_TRACKED,
            provenance_columns=(
                "age_provenance",
                "weight_kg_provenance",
                "bmi_provenance",
                *DEFAULT_FEATURE_PROVENANCE_COLUMNS,
            ),
        ),
    ),
    inferred_fields=(
        FeatureSpec(
            feature_name="frailty_surrogate_score",
            description="Future frailty surrogate placeholder derived from safe reserve proxies.",
            family=FeatureFamily.PATIENT_CONTEXT,
            derivation_type=FeatureDerivationType.INFERRED,
            source_tables=("patients", "omr", "admissions", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=("frailty_surrogates_definition_version", *DEFAULT_FEATURE_PROVENANCE_COLUMNS),
        ),
    ),
)


PRIOR_UTILIZATION_SCHEMA = FeatureGroupSchema(
    group_name="prior_utilization",
    family=FeatureFamily.PATIENT_CONTEXT,
    purpose="Placeholder schema for encounter history available before the current review timestamp.",
    transformed_fields=(
        FeatureSpec(
            feature_name="prior_encounter_count_lookback",
            description="Future pre-review encounter count within a governed lookback window.",
            family=FeatureFamily.PATIENT_CONTEXT,
            derivation_type=FeatureDerivationType.TRANSFORMED,
            source_tables=("admissions", "edstays", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=("prior_utilization_window_definition", *DEFAULT_FEATURE_PROVENANCE_COLUMNS),
        ),
    ),
    inferred_fields=(
        FeatureSpec(
            feature_name="recent_utilization_pattern",
            description="Future inferred utilization pattern placeholder constrained to pre-review encounters.",
            family=FeatureFamily.PATIENT_CONTEXT,
            derivation_type=FeatureDerivationType.INFERRED,
            source_tables=("admissions", "edstays", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=("prior_utilization_window_definition", *DEFAULT_FEATURE_PROVENANCE_COLUMNS),
        ),
    ),
)


LAB_TRAJECTORY_WINDOWS_SCHEMA = FeatureGroupSchema(
    group_name="lab_trajectory_windows",
    family=FeatureFamily.TEMPORAL_PHYSIOLOGY,
    purpose="Placeholder schema for governed lab trajectory windows that end at or before review time.",
    direct_fields=(
        FeatureSpec(
            feature_name="lab_window_bounds_recorded_flag",
            description="Explicit flag that lab trajectory window bounds were materialized and validated.",
            family=FeatureFamily.TEMPORAL_PHYSIOLOGY,
            derivation_type=FeatureDerivationType.DIRECT,
            source_tables=("labevents", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.MUST_BE_EXPLICITLY_TRACKED,
            provenance_columns=("lab_window_start", "lab_window_end", *DEFAULT_FEATURE_PROVENANCE_COLUMNS),
        ),
    ),
    transformed_fields=(
        FeatureSpec(
            feature_name="creatinine_trajectory_window_summary",
            description="Future creatinine trajectory window placeholder.",
            family=FeatureFamily.TEMPORAL_PHYSIOLOGY,
            derivation_type=FeatureDerivationType.TRANSFORMED,
            source_tables=("labevents", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=("creatinine_provenance", "lab_window_end", *DEFAULT_FEATURE_PROVENANCE_COLUMNS),
        ),
    ),
    inferred_fields=(
        FeatureSpec(
            feature_name="lab_trajectory_instability_flag",
            description="Future inferred lab instability placeholder from time-safe windows.",
            family=FeatureFamily.TEMPORAL_PHYSIOLOGY,
            derivation_type=FeatureDerivationType.INFERRED,
            source_tables=("labevents", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=("lab_window_end", *DEFAULT_FEATURE_PROVENANCE_COLUMNS),
        ),
    ),
)


VITAL_TRAJECTORY_WINDOWS_SCHEMA = FeatureGroupSchema(
    group_name="vital_trajectory_windows",
    family=FeatureFamily.TEMPORAL_PHYSIOLOGY,
    purpose="Placeholder schema for governed vital-sign trajectory windows that end at or before review time.",
    direct_fields=(
        FeatureSpec(
            feature_name="vital_window_bounds_recorded_flag",
            description="Explicit flag that vital trajectory window bounds were materialized and validated.",
            family=FeatureFamily.TEMPORAL_PHYSIOLOGY,
            derivation_type=FeatureDerivationType.DIRECT,
            source_tables=("triage", "vitalsign", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.MUST_BE_EXPLICITLY_TRACKED,
            provenance_columns=("vital_window_start", "vital_window_end", *DEFAULT_FEATURE_PROVENANCE_COLUMNS),
        ),
    ),
    transformed_fields=(
        FeatureSpec(
            feature_name="hemodynamic_trajectory_window_summary",
            description="Future hemodynamic trajectory window placeholder.",
            family=FeatureFamily.TEMPORAL_PHYSIOLOGY,
            derivation_type=FeatureDerivationType.TRANSFORMED,
            source_tables=("triage", "vitalsign", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=("blood_pressure_provenance", "heart_rate_provenance", "vital_window_end", *DEFAULT_FEATURE_PROVENANCE_COLUMNS),
        ),
    ),
    inferred_fields=(
        FeatureSpec(
            feature_name="vital_instability_flag",
            description="Future inferred vital instability placeholder from time-safe windows.",
            family=FeatureFamily.TEMPORAL_PHYSIOLOGY,
            derivation_type=FeatureDerivationType.INFERRED,
            source_tables=("triage", "vitalsign", "encounter_medication_state"),
            point_in_time_safe=True,
            expected_missingness_behavior=ExpectedMissingnessBehavior.STRUCTURAL_PLACEHOLDER_UNTIL_IMPLEMENTED,
            provenance_columns=("vital_window_end", *DEFAULT_FEATURE_PROVENANCE_COLUMNS),
        ),
    ),
)


FEATURE_GROUP_SCHEMAS: dict[str, FeatureGroupSchema] = {
    group.group_name: group
    for group in (
        MEDICATION_SEMANTICS_FEATURE_GROUP,
        MEDICATION_BURDEN_FEATURE_GROUP,
        PATIENT_CONTEXT_FEATURE_GROUP,
        TEMPORAL_PHYSIOLOGY_FEATURE_GROUP,
        PROVENANCE_MISSINGNESS_FEATURE_GROUP,
        CLASS_BURDEN_COUNTS_SCHEMA,
        RENAL_VULNERABILITY_SCHEMA,
        FRAILTY_SURROGATES_SCHEMA,
        PRIOR_UTILIZATION_SCHEMA,
        LAB_TRAJECTORY_WINDOWS_SCHEMA,
        VITAL_TRAJECTORY_WINDOWS_SCHEMA,
    )
}


def iter_all_feature_specs() -> tuple[FeatureSpec, ...]:
    """Return all registered feature specs across all groups."""
    return tuple(
        feature
        for group in FEATURE_GROUP_SCHEMAS.values()
        for feature in group.all_fields()
    )


def validate_feature_group_schema(group: FeatureGroupSchema) -> None:
    """Validate schema integrity for one feature group."""
    if not group.group_name.strip():
        raise ValueError("Feature group name must be non-empty.")
    if not group.purpose.strip():
        raise ValueError(f"Feature group '{group.group_name}' must declare a purpose.")

    seen_names: set[str] = set()
    for expected_derivation, field_specs in [
        (FeatureDerivationType.DIRECT, group.direct_fields),
        (FeatureDerivationType.TRANSFORMED, group.transformed_fields),
        (FeatureDerivationType.INFERRED, group.inferred_fields),
    ]:
        for spec in field_specs:
            if spec.family != group.family:
                raise ValueError(
                    f"Feature '{spec.feature_name}' has family '{spec.family.value}' "
                    f"but is registered under '{group.family.value}'."
                )
            if spec.derivation_type != expected_derivation:
                raise ValueError(
                    f"Feature '{spec.feature_name}' is placed under '{expected_derivation.value}' "
                    f"but declares '{spec.derivation_type.value}'."
                )
            if not spec.feature_name.strip():
                raise ValueError(f"Feature in group '{group.group_name}' must have a non-empty name.")
            if spec.feature_name in seen_names:
                raise ValueError(
                    f"Feature group '{group.group_name}' contains duplicate feature '{spec.feature_name}'."
                )
            if not spec.description.strip():
                raise ValueError(f"Feature '{spec.feature_name}' must declare a description.")
            if not spec.source_tables:
                raise ValueError(f"Feature '{spec.feature_name}' must declare source tables.")
            if not spec.provenance_columns:
                raise ValueError(f"Feature '{spec.feature_name}' must declare provenance columns.")
            if not spec.point_in_time_safe:
                raise ValueError(
                    f"Feature '{spec.feature_name}' is not point-in-time safe and cannot be registered."
                )
            seen_names.add(spec.feature_name)


def validate_feature_registry() -> None:
    """Validate integrity across the full registered feature schema set."""
    global_names: set[str] = set()
    for group in FEATURE_GROUP_SCHEMAS.values():
        validate_feature_group_schema(group)
        for spec in group.all_fields():
            if spec.feature_name in global_names:
                raise ValueError(
                    f"Duplicate feature name '{spec.feature_name}' found across feature groups."
                )
            global_names.add(spec.feature_name)
