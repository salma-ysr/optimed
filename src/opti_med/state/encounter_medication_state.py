"""Software interface for the future encounter-medication-state layer.

This module defines the stable target shape for future builders that emit one
row per review-time medication state. It is intentionally not wired to any
loader, CSV, or current pipeline artifact yet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Literal

from opti_med.contracts.pipeline import ANALYTICAL_GRAIN_DESCRIPTION, ProvenanceMap


# Keep this type-level literal set aligned with the shared runtime labels in
# `opti_med.time_semantics.constants`.
MedicationStatusAtReview = Literal[
    "active_at_review_time",
    "inactive_before_review_time",
    "pre_admission_only",
    "activity_uncertain_at_review_time",
]


ENCOUNTER_MEDICATION_STATE_GRAIN_DESCRIPTION = ANALYTICAL_GRAIN_DESCRIPTION
ENCOUNTER_MEDICATION_STATE_PRIMARY_KEY = (
    "subject_id",
    "encounter_id",
    "medication_standardized",
    "review_timestamp",
)


@dataclass(frozen=True, slots=True)
class EncounterMedicationStateKey:
    """Future analytical key for one review-time medication state row.

    The intended long-term key is:
    subject_id + encounter_id + medication_standardized + review_timestamp

    `medication_standardized` remains nullable here only because the repository
    has not implemented medication standardization yet. Future builders should
    treat it as a required field once that layer exists.
    """

    subject_id: int
    encounter_id: str
    medication_standardized: str | None
    review_timestamp: datetime


@dataclass(frozen=True, slots=True)
class EncounterMedicationStateRow:
    """One review-time medication row at the future modeling grain.

    This is intentionally not a cohort-episode row and not a collapsed
    prescription interval row. It represents the medication state known at a
    specific review timestamp for a specific encounter.
    """

    subject_id: int
    encounter_id: str
    review_timestamp: datetime
    hadm_id: int | None = None
    stay_id: int | None = None
    review_timestamp_source: str | None = None
    medication_raw: str | None = None
    medication_normalized: str | None = None
    medication_standardized: str | None = None
    medication_standardized_source: str | None = None
    rxnorm_rxcui: str | None = None
    ingredient_standardized: str | None = None
    ingredient_resolution_status: str | None = None
    mapping_confidence: str | None = None
    ambiguous_mapping_flag: bool = False
    # TODO(ml-pivot): populate once medication class standardization exists.
    medication_class_standardized: str | None = None
    medication_status_at_review: MedicationStatusAtReview | None = None
    active_at_review_flag: bool = False
    continued_from_home_inferred: bool = False
    newly_started_during_encounter_inferred: bool = False
    route: str | None = None
    frequency: str | None = None
    status: str | None = None
    source_home_medrecon_flag: bool = False
    source_ed_pyxis_flag: bool = False
    source_hospital_order_flag: bool = False
    source_hospital_admin_flag: bool = False
    provenance: ProvenanceMap = field(default_factory=dict)

    def to_key(self) -> EncounterMedicationStateKey:
        """Return the future analytical key for this review-time medication row."""
        return EncounterMedicationStateKey(
            subject_id=self.subject_id,
            encounter_id=self.encounter_id,
            medication_standardized=self.medication_standardized,
            review_timestamp=self.review_timestamp,
        )


class EncounterMedicationStateBuilder(ABC):
    """Abstract producer interface for future encounter-medication-state builders."""

    @property
    def output_grain_description(self) -> str:
        """Return the intended analytical grain for emitted rows."""
        return ENCOUNTER_MEDICATION_STATE_GRAIN_DESCRIPTION

    @property
    def primary_key_fields(self) -> tuple[str, ...]:
        """Return the intended primary key field names for emitted rows."""
        return ENCOUNTER_MEDICATION_STATE_PRIMARY_KEY

    @abstractmethod
    def build_rows(self) -> Iterable[EncounterMedicationStateRow]:
        """Build review-time medication state rows.

        Implementations must emit one row per review-time medication state, not
        one row per cohort episode, prescription segment, or scored row.
        """
        raise NotImplementedError
