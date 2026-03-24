"""Baseline medication semantics adapter.

This adapter is deliberately conservative. It only wraps the repository's
current string normalization logic and returns clearly marked placeholder
outputs for ingredient identity, class membership, PRN detection, and dose
normalization.
"""

from __future__ import annotations

from opti_med.data_access.medication_consolidation import normalize_medication_name
from opti_med.medication_semantics.contracts import (
    MedicationClassIdentity,
    MedicationScheduleSemantics,
    MedicationSemanticMapper,
    NormalizedDose,
    NormalizedMedicationName,
    StandardizedMedicationIdentity,
)


class BaselineTextMedicationSemanticMapper(MedicationSemanticMapper):
    """Placeholder adapter that preserves current text normalization behavior.

    This is not ontology-backed mapping. It should not be mistaken for true
    ingredient standardization or clinically validated class membership.
    """

    mapper_name = "baseline_text_medication_semantics"
    mapper_version = "0.1"

    def normalize_name(self, raw_name: object) -> NormalizedMedicationName:
        normalized_text = normalize_medication_name(raw_name)
        return NormalizedMedicationName(
            raw_name=None if raw_name is None else str(raw_name),
            normalized_text=normalized_text,
            normalizer_name=self.mapper_name,
            normalizer_version=self.mapper_version,
            normalization_status=(
                "placeholder_text_only" if normalized_text is not None else "not_standardized"
            ),
            provenance={
                "baseline_only": True,
                "semantic_quality": "text_normalized_only",
            },
        )

    def standardize_ingredient(
        self,
        normalized_name: NormalizedMedicationName | str | None,
    ) -> StandardizedMedicationIdentity:
        normalized_text = _normalized_text_value(normalized_name)
        return StandardizedMedicationIdentity(
            normalized_text=normalized_text,
            standardized_ingredient_id=None,
            standardized_ingredient_label=None,
            identity_system=None,
            identity_system_version=None,
            standardization_status=(
                "placeholder_text_only" if normalized_text is not None else "not_standardized"
            ),
            provenance={
                "baseline_only": True,
                "semantic_quality": "no_true_standardized_identity",
            },
        )

    def infer_class(
        self,
        standardized_ingredient: StandardizedMedicationIdentity | str | None,
    ) -> tuple[MedicationClassIdentity, ...]:
        identity_text = _standardized_identity_text(standardized_ingredient)
        if identity_text is None:
            return tuple()
        return tuple()

    def detect_prn_vs_scheduled(
        self,
        *,
        frequency: str | None = None,
        status: str | None = None,
        route: str | None = None,
    ) -> MedicationScheduleSemantics:
        del frequency, status, route
        return "unknown"

    def normalize_dose(
        self,
        *,
        dose_value: object = None,
        dose_unit: str | None = None,
        normalized_name: NormalizedMedicationName | None = None,
        standardized_ingredient: StandardizedMedicationIdentity | None = None,
    ) -> NormalizedDose:
        del normalized_name, standardized_ingredient
        return NormalizedDose(
            input_dose_value=None if dose_value is None else str(dose_value),
            input_dose_unit=dose_unit,
            normalized_dose_value=None,
            normalized_dose_unit=None,
            normalization_status="not_standardized",
            provenance={
                "baseline_only": True,
                "semantic_quality": "no_dose_normalization",
            },
        )


def _normalized_text_value(value: NormalizedMedicationName | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, NormalizedMedicationName):
        return value.normalized_text
    text = str(value).strip()
    return text or None


def _standardized_identity_text(
    value: StandardizedMedicationIdentity | str | None,
) -> str | None:
    if value is None:
        return None
    if isinstance(value, StandardizedMedicationIdentity):
        return value.standardized_ingredient_label or value.normalized_text
    text = str(value).strip()
    return text or None
