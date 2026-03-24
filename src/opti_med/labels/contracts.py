"""Label strategy contracts for future OPTI-MED model development.

These contracts are intentionally decision-focused. They define what labels
future builders should target and which shortcuts are explicitly forbidden
before any label extraction work begins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from opti_med.contracts.pipeline import (
    ANALYTICAL_GRAIN_DESCRIPTION,
    ANALYTICAL_GRAIN_PRIMARY_KEY,
    ProvenanceMap,
)


ANALYTICAL_LABEL_GRAIN_DESCRIPTION = ANALYTICAL_GRAIN_DESCRIPTION
ANALYTICAL_LABEL_PRIMARY_KEY = ANALYTICAL_GRAIN_PRIMARY_KEY

PRIMARY_ACTION_LABEL_NAME = "primary_action_label"
AUXILIARY_HARM_LABEL_NAME = "auxiliary_harm_labels"
EXCLUSION_FLAG_NAME = "exclusion_flags"

PROHIBITED_TRAINING_LABELS: tuple[str, ...] = (
    "deprescribing_priority_score",
    "deprescribing_priority_label",
    "deprescribing_priority_summary_alert",
    "deprescribing_priority_explanation",
    "current_rule_score",
    "current_rule_label",
)


class PrimaryActionLabelCategory(str, Enum):
    """Candidate primary action labels for future supervised learning."""

    STOPPED_OR_DEINTENSIFIED_BEFORE_DISCHARGE = (
        "stopped_or_deintensified_before_discharge"
    )
    NO_CLEAR_STOP_OR_DEINTENSIFICATION_BEFORE_DISCHARGE = (
        "no_clear_stop_or_deintensification_before_discharge"
    )
    ACTION_UNDETERMINED = "action_undetermined"
    WINDOW_CENSORED = "window_censored"


class AuxiliaryHarmLabelCategory(str, Enum):
    """Candidate auxiliary harm labels anchored after review time."""

    RENAL_DETERIORATION = "renal_deterioration"
    HEMODYNAMIC_INSTABILITY = "hemodynamic_instability"
    ELECTROLYTE_INSTABILITY = "electrolyte_instability"
    OVERSEDATION_RESPIRATORY_RISK = "oversedation_respiratory_risk"


class ExclusionCategory(str, Enum):
    """Medication categories to exclude or downweight in future training."""

    ACUTE_LIFE_SUSTAINING_MEDICATION = "acute_life_sustaining_medication"
    FUNDAMENTALLY_DIFFERENT_DEPRESCRIBING_LOGIC = (
        "fundamentally_different_deprescribing_logic"
    )


class LabelSourceCategory(str, Enum):
    """High-level provenance categories for constructed labels."""

    PRESCRIPTION_CHANGE_EVIDENCE = "prescription_change_evidence"
    ADMINISTRATION_EVIDENCE = "administration_evidence"
    LAB_EVIDENCE = "lab_evidence"
    VITAL_EVIDENCE = "vital_evidence"
    CLINICAL_CONTEXT_EVIDENCE = "clinical_context_evidence"
    CONSTRUCTED_LABEL_LOGIC = "constructed_label_logic"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True, slots=True)
class LabelWindowMetadata:
    """Observation window metadata for label construction."""

    review_timestamp: datetime
    action_window_start: datetime
    action_window_end: datetime | None
    harm_window_start: datetime
    harm_window_end: datetime | None
    discharge_timestamp: datetime | None = None
    window_censored_flag: bool = False
    window_definition_version: str = "0.1"


@dataclass(frozen=True, slots=True)
class LabelProvenanceMetadata:
    """Provenance metadata for future constructed labels."""

    label_definition_version: str
    source_categories: tuple[LabelSourceCategory, ...]
    source_tables: tuple[str, ...]
    constructed_label_flag: bool = True
    rule_score_used_as_training_label: bool = False
    notes: str | None = None
    provenance: ProvenanceMap = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PrimaryActionLabelRow:
    """Candidate primary action label row at the review-time medication grain."""

    subject_id: int
    encounter_id: str
    review_timestamp: datetime
    medication_standardized: str | None
    primary_action_label: PrimaryActionLabelCategory
    hadm_id: int | None = None
    stay_id: int | None = None
    medication_normalized: str | None = None
    window: LabelWindowMetadata | None = None
    provenance: LabelProvenanceMetadata | None = None


@dataclass(frozen=True, slots=True)
class AuxiliaryHarmLabelsRow:
    """Candidate auxiliary harm label row at the review-time medication grain."""

    subject_id: int
    encounter_id: str
    review_timestamp: datetime
    medication_standardized: str | None
    renal_deterioration: bool | None = None
    hemodynamic_instability: bool | None = None
    electrolyte_instability: bool | None = None
    oversedation_respiratory_risk: bool | None = None
    hadm_id: int | None = None
    stay_id: int | None = None
    window: LabelWindowMetadata | None = None
    provenance: LabelProvenanceMetadata | None = None

    @staticmethod
    def supported_categories() -> tuple[AuxiliaryHarmLabelCategory, ...]:
        """Return the supported auxiliary harm categories in stable order."""
        return (
            AuxiliaryHarmLabelCategory.RENAL_DETERIORATION,
            AuxiliaryHarmLabelCategory.HEMODYNAMIC_INSTABILITY,
            AuxiliaryHarmLabelCategory.ELECTROLYTE_INSTABILITY,
            AuxiliaryHarmLabelCategory.OVERSEDATION_RESPIRATORY_RISK,
        )


@dataclass(frozen=True, slots=True)
class ExclusionFlagsRow:
    """Exclusion and downweighting flags at the review-time medication grain."""

    subject_id: int
    encounter_id: str
    review_timestamp: datetime
    medication_standardized: str | None
    acute_life_sustaining_medication: bool = False
    fundamentally_different_deprescribing_logic: bool = False
    excluded_from_primary_training: bool = False
    downweight_recommended: bool = False
    hadm_id: int | None = None
    stay_id: int | None = None
    window: LabelWindowMetadata | None = None
    provenance: LabelProvenanceMetadata | None = None

    @staticmethod
    def supported_categories() -> tuple[ExclusionCategory, ...]:
        """Return the supported exclusion categories in stable order."""
        return (
            ExclusionCategory.ACUTE_LIFE_SUSTAINING_MEDICATION,
            ExclusionCategory.FUNDAMENTALLY_DIFFERENT_DEPRESCRIBING_LOGIC,
        )
