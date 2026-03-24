"""Medication semantics contracts for future ontology-backed representation.

This package is intentionally separate from current feature and scoring code so
the repository can distinguish:

- normalized text
- standardized ingredient identity
- class membership

No real external vocabulary is loaded here yet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from opti_med.contracts.pipeline import ProvenanceMap


MedicationScheduleSemantics = Literal["scheduled", "prn", "unknown"]
StandardizationStatus = Literal[
    "not_standardized",
    "placeholder_text_only",
    "keyword_placeholder",
    "ontology_backed",
]


FUTURE_BURDEN_FEATURE_TODOS: tuple[str, ...] = (
    "opioid_mme",
    "sedative_burden",
    "anticholinergic_burden",
    "renal_dose_mismatch",
    "duplicate_therapy",
)


@dataclass(frozen=True, slots=True)
class NormalizedMedicationName:
    """Text normalization output.

    This contract is only about normalized text representation. It is not a
    standardized ingredient identity and it is not a medication class.
    """

    raw_name: str | None
    normalized_text: str | None
    normalizer_name: str
    normalizer_version: str
    normalization_status: StandardizationStatus
    provenance: ProvenanceMap = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StandardizedMedicationIdentity:
    """Standardized ingredient-level identity.

    This contract is intentionally separate from normalized text. A row can
    have normalized text without having a true standardized identity yet.
    """

    normalized_text: str | None
    standardized_ingredient_id: str | None
    standardized_ingredient_label: str | None
    identity_system: str | None
    identity_system_version: str | None
    standardization_status: StandardizationStatus
    provenance: ProvenanceMap = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MedicationClassIdentity:
    """Medication class membership identity.

    This contract is intentionally separate from both normalized text and
    standardized ingredient identity. Future ontology-backed class membership
    should attach to a standardized ingredient, not to ad hoc raw strings.
    """

    standardized_ingredient_id: str | None
    class_id: str | None
    class_label: str | None
    class_system: str | None
    class_system_version: str | None
    membership_status: StandardizationStatus
    provenance: ProvenanceMap = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NormalizedDose:
    """Dose normalization output.

    This remains a placeholder contract until real dose normalization rules or
    vocabulary-backed units are introduced.
    """

    input_dose_value: str | None
    input_dose_unit: str | None
    normalized_dose_value: float | None
    normalized_dose_unit: str | None
    normalization_status: StandardizationStatus
    provenance: ProvenanceMap = field(default_factory=dict)


class MedicationSemanticMapper(ABC):
    """Abstract mapper for future ontology-backed medication semantics."""

    mapper_name: str
    mapper_version: str

    @abstractmethod
    def normalize_name(self, raw_name: object) -> NormalizedMedicationName:
        """Normalize a raw medication string into comparable text."""
        raise NotImplementedError

    @abstractmethod
    def standardize_ingredient(
        self,
        normalized_name: NormalizedMedicationName | str | None,
    ) -> StandardizedMedicationIdentity:
        """Map normalized text to a standardized ingredient identity."""
        raise NotImplementedError

    @abstractmethod
    def infer_class(
        self,
        standardized_ingredient: StandardizedMedicationIdentity | str | None,
    ) -> tuple[MedicationClassIdentity, ...]:
        """Infer medication class membership from a standardized ingredient identity."""
        raise NotImplementedError

    @abstractmethod
    def detect_prn_vs_scheduled(
        self,
        *,
        frequency: str | None = None,
        status: str | None = None,
        route: str | None = None,
    ) -> MedicationScheduleSemantics:
        """Infer scheduled-vs-PRN semantics from structured medication context."""
        raise NotImplementedError

    @abstractmethod
    def normalize_dose(
        self,
        *,
        dose_value: object = None,
        dose_unit: str | None = None,
        normalized_name: NormalizedMedicationName | None = None,
        standardized_ingredient: StandardizedMedicationIdentity | None = None,
    ) -> NormalizedDose:
        """Normalize dose value and unit to a future standard representation."""
        raise NotImplementedError
