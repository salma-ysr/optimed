"""Explicit contracts for the future staged analytical pipeline.

These contracts are intentionally implementation-light. They define the target
interfaces, keys, provenance rules, and point-in-time expectations before any
full-data ingestion or ML-specific pipeline work begins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal, TypeAlias


ScalarValue: TypeAlias = str | int | float | bool | None
ProvenanceValue: TypeAlias = ScalarValue
ProvenanceMap: TypeAlias = dict[str, ProvenanceValue]
ScheduledVsPrn: TypeAlias = Literal["scheduled", "prn", "unknown"]


ANALYTICAL_GRAIN_DESCRIPTION = (
    "one row per subject_id + encounter_id + medication_standardized + review_timestamp"
)
ANALYTICAL_GRAIN_PRIMARY_KEY = (
    "subject_id",
    "encounter_id",
    "medication_standardized",
    "review_timestamp",
)


class MissingnessExpectation(str, Enum):
    """Allowed missingness pattern for a contract."""

    REFERENCE_METADATA_OPTIONAL = (
        "reference identifiers required; descriptive metadata nullable with provenance"
    )
    STANDARDIZED_NON_KEY_FIELDS_NULLABLE = (
        "primary key required; non-key fields may be null when provenance explains why"
    )
    REVIEW_TIME_LAYER_BOOTSTRAP_NULLS_ALLOWED = (
        "primary key and review timestamp required; bootstrap placeholder fields may be null"
    )
    FEATURE_VALUES_PARTIAL_WITH_FLAGS = (
        "primary key required; feature payload may be partial and must expose availability flags"
    )
    LABEL_VALUES_NULLABLE_WITH_WINDOW_REASON = (
        "primary key required; labels may be null because of censoring, pending observation, or exclusions"
    )
    MODEL_OUTPUTS_NULLABLE_ONLY_IF_SCORING_SKIPPED = (
        "primary key required; prediction fields may be null only when scoring is intentionally skipped"
    )
    GUIDANCE_TEXT_NULLABLE_WITH_SUPPRESSION_REASON = (
        "primary key required; guidance may be suppressed but suppression reason must be present"
    )


class PointInTimeConstraint(str, Enum):
    """Whether and how a layer is time-bounded."""

    NOT_POINT_IN_TIME_CONSTRAINED = "not point-in-time constrained"
    SNAPSHOT_CONSTRAINED = "constrained to a source snapshot or build snapshot"
    AS_OF_REVIEW_TIME = "must be valid strictly as of review_timestamp"
    POST_REVIEW_OUTCOME_WINDOW = "anchored to review_timestamp with a post-review outcome window"
    DERIVED_FROM_PRIOR_PIT_LAYERS = "derived from point-in-time constrained upstream layers"


@dataclass(frozen=True, slots=True)
class ContractSpec:
    """Human-readable metadata for one pipeline contract."""

    name: str
    purpose: str
    primary_key: tuple[str, ...]
    allowed_provenance_fields: tuple[str, ...]
    missingness_expectation: MissingnessExpectation
    point_in_time_constraint: PointInTimeConstraint


@dataclass(frozen=True, slots=True)
class RawTableReference:
    """Reference to an immutable source table before standardization."""

    source_system: str
    raw_table_name: str
    source_snapshot_id: str
    artifact_uri: str
    file_format: str | None = None
    extract_run_id: str | None = None
    schema_version: str | None = None
    provenance: ProvenanceMap = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StandardizedTableReference:
    """Reference to a canonicalized table produced from raw inputs."""

    standardized_table_name: str
    standardized_snapshot_id: str
    artifact_uri: str
    build_run_id: str
    upstream_raw_table_ids: tuple[str, ...] = ()
    schema_version: str | None = None
    provenance: ProvenanceMap = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EncounterMedicationStateRow:
    """Canonical review-time state for one medication in one encounter."""

    subject_id: int
    encounter_id: str
    review_timestamp: datetime
    # TODO(ml-pivot): make this required once medication standardization exists.
    medication_standardized: str | None = None
    hadm_id: int | None = None
    stay_id: int | None = None
    medication_observed_name: str | None = None
    # TODO(ml-pivot): populate once medication classes are standardized.
    medication_class_standardized: str | None = None
    # TODO(ml-pivot): populate once scheduled/PRN semantics are standardized.
    scheduled_vs_prn: ScheduledVsPrn | None = None
    # TODO(ml-pivot): populate once dose normalization rules exist.
    dose_normalized: float | None = None
    dose_normalized_unit: str | None = None
    medication_status: str | None = None
    review_timestamp_source: str | None = None
    # TODO(ml-pivot): set once review-time validation logic exists.
    review_time_validated_flag: bool | None = None
    provenance: ProvenanceMap = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """Point-in-time feature store row at the encounter-medication review grain."""

    subject_id: int
    encounter_id: str
    review_timestamp: datetime
    # TODO(ml-pivot): make this required once medication standardization exists.
    medication_standardized: str | None = None
    feature_set_version: str | None = None
    feature_values: dict[str, ScalarValue] = field(default_factory=dict)
    # TODO(ml-pivot): expose once medication classes are standardized.
    medication_class_standardized: str | None = None
    # TODO(ml-pivot): expose once scheduled/PRN semantics are standardized.
    scheduled_vs_prn: ScheduledVsPrn | None = None
    # TODO(ml-pivot): expose once dose normalization rules exist.
    dose_normalized: float | None = None
    # TODO(ml-pivot): set once review-time validation logic exists.
    review_time_validated_flag: bool | None = None
    # TODO(ml-pivot): set once feature availability auditing exists.
    feature_available_as_of_review_time_flag: bool | None = None
    provenance: ProvenanceMap = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LabelRow:
    """Outcome label row aligned to the encounter-medication review grain."""

    subject_id: int
    encounter_id: str
    review_timestamp: datetime
    # TODO(ml-pivot): make this required once medication standardization exists.
    medication_standardized: str | None = None
    label_name: str = "unspecified_label"
    label_value: ScalarValue = None
    label_observed_timestamp: datetime | None = None
    label_window_start: datetime | None = None
    label_window_end: datetime | None = None
    label_valid_flag: bool | None = None
    censoring_reason: str | None = None
    provenance: ProvenanceMap = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelOutputRow:
    """Prediction output row aligned to the encounter-medication review grain."""

    subject_id: int
    encounter_id: str
    review_timestamp: datetime
    model_name: str
    model_version: str
    model_run_id: str
    # TODO(ml-pivot): make this required once medication standardization exists.
    medication_standardized: str | None = None
    predicted_label: str | None = None
    predicted_score: float | None = None
    predicted_probability: float | None = None
    score_timestamp: datetime | None = None
    scoring_skipped_reason: str | None = None
    provenance: ProvenanceMap = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GuidanceRow:
    """Human-consumable guidance row derived from model and/or rules."""

    subject_id: int
    encounter_id: str
    review_timestamp: datetime
    guidance_version: str
    # TODO(ml-pivot): make this required once medication standardization exists.
    medication_standardized: str | None = None
    recommended_action: str | None = None
    guidance_text: str | None = None
    guidance_priority: str | None = None
    requires_human_review: bool = True
    suppression_reason: str | None = None
    provenance: ProvenanceMap = field(default_factory=dict)


RAW_TABLE_REFERENCE_SPEC = ContractSpec(
    name="RawTableReference",
    purpose=(
        "Declare immutable upstream source-table artifacts before any standardization or feature logic."
    ),
    primary_key=("source_system", "raw_table_name", "source_snapshot_id"),
    allowed_provenance_fields=(
        "source_system",
        "raw_table_name",
        "source_snapshot_id",
        "extract_run_id",
        "file_format",
        "schema_version",
        "source_checksum",
        "source_row_count",
        "source_last_modified_at",
    ),
    missingness_expectation=MissingnessExpectation.REFERENCE_METADATA_OPTIONAL,
    point_in_time_constraint=PointInTimeConstraint.SNAPSHOT_CONSTRAINED,
)


STANDARDIZED_TABLE_REFERENCE_SPEC = ContractSpec(
    name="StandardizedTableReference",
    purpose=(
        "Declare canonicalized table artifacts that preserve lineage back to raw inputs."
    ),
    primary_key=("standardized_table_name", "standardized_snapshot_id"),
    allowed_provenance_fields=(
        "standardized_table_name",
        "standardized_snapshot_id",
        "build_run_id",
        "schema_version",
        "upstream_raw_table_ids",
        "standardization_version",
        "record_count",
        "build_timestamp",
    ),
    missingness_expectation=MissingnessExpectation.STANDARDIZED_NON_KEY_FIELDS_NULLABLE,
    point_in_time_constraint=PointInTimeConstraint.SNAPSHOT_CONSTRAINED,
)


ENCOUNTER_MEDICATION_STATE_SPEC = ContractSpec(
    name="EncounterMedicationStateRow",
    purpose=(
        "Represent one medication state at the future analytical grain of "
        f"{ANALYTICAL_GRAIN_DESCRIPTION}."
    ),
    primary_key=ANALYTICAL_GRAIN_PRIMARY_KEY,
    allowed_provenance_fields=(
        "encounter_state_run_id",
        "review_timestamp_source",
        "review_timestamp_source_priority",
        "source_home_medrecon",
        "source_ed_pyxis",
        "source_hospital_order",
        "source_hospital_admin",
        "continued_from_home_inferred",
        "newly_started_during_encounter_inferred",
        "state_contract_version",
    ),
    missingness_expectation=MissingnessExpectation.REVIEW_TIME_LAYER_BOOTSTRAP_NULLS_ALLOWED,
    point_in_time_constraint=PointInTimeConstraint.AS_OF_REVIEW_TIME,
)


FEATURE_ROW_SPEC = ContractSpec(
    name="FeatureRow",
    purpose=(
        "Store ML-ready features aligned to the encounter-medication review grain without leaking post-review information."
    ),
    primary_key=ANALYTICAL_GRAIN_PRIMARY_KEY,
    allowed_provenance_fields=(
        "feature_run_id",
        "feature_set_version",
        "feature_window_definition",
        "feature_contract_version",
        "feature_available_as_of_review_time_flag",
        "upstream_state_contract_version",
        "source_table_lineage",
    ),
    missingness_expectation=MissingnessExpectation.FEATURE_VALUES_PARTIAL_WITH_FLAGS,
    point_in_time_constraint=PointInTimeConstraint.AS_OF_REVIEW_TIME,
)


LABEL_ROW_SPEC = ContractSpec(
    name="LabelRow",
    purpose=(
        "Store supervised outcome labels at the encounter-medication review grain with explicit observation windows."
    ),
    primary_key=ANALYTICAL_GRAIN_PRIMARY_KEY,
    allowed_provenance_fields=(
        "label_run_id",
        "label_definition_version",
        "label_window_type",
        "label_window_start",
        "label_window_end",
        "adjudication_source",
        "censoring_reason",
        "label_contract_version",
    ),
    missingness_expectation=MissingnessExpectation.LABEL_VALUES_NULLABLE_WITH_WINDOW_REASON,
    point_in_time_constraint=PointInTimeConstraint.POST_REVIEW_OUTCOME_WINDOW,
)


MODEL_OUTPUT_ROW_SPEC = ContractSpec(
    name="ModelOutputRow",
    purpose=(
        "Store model predictions at the encounter-medication review grain for offline evaluation or future serving."
    ),
    primary_key=ANALYTICAL_GRAIN_PRIMARY_KEY,
    allowed_provenance_fields=(
        "model_name",
        "model_version",
        "model_run_id",
        "model_registry_id",
        "feature_set_version",
        "score_timestamp",
        "calibration_version",
        "scoring_skipped_reason",
    ),
    missingness_expectation=MissingnessExpectation.MODEL_OUTPUTS_NULLABLE_ONLY_IF_SCORING_SKIPPED,
    point_in_time_constraint=PointInTimeConstraint.DERIVED_FROM_PRIOR_PIT_LAYERS,
)


GUIDANCE_ROW_SPEC = ContractSpec(
    name="GuidanceRow",
    purpose=(
        "Store clinician-facing guidance derived from model outputs, baseline rules, or both at the encounter-medication review grain."
    ),
    primary_key=ANALYTICAL_GRAIN_PRIMARY_KEY,
    allowed_provenance_fields=(
        "guidance_run_id",
        "guidance_policy_version",
        "guidance_generated_at",
        "based_on_model_run_id",
        "based_on_rule_baseline_version",
        "suppression_reason",
        "requires_human_review",
    ),
    missingness_expectation=MissingnessExpectation.GUIDANCE_TEXT_NULLABLE_WITH_SUPPRESSION_REASON,
    point_in_time_constraint=PointInTimeConstraint.DERIVED_FROM_PRIOR_PIT_LAYERS,
)


CONTRACT_SPECS: dict[str, ContractSpec] = {
    spec.name: spec
    for spec in (
        RAW_TABLE_REFERENCE_SPEC,
        STANDARDIZED_TABLE_REFERENCE_SPEC,
        ENCOUNTER_MEDICATION_STATE_SPEC,
        FEATURE_ROW_SPEC,
        LABEL_ROW_SPEC,
        MODEL_OUTPUT_ROW_SPEC,
        GUIDANCE_ROW_SPEC,
    )
}
