"""First-pass medication class enrichment keyed to RxNorm-standardized ingredients."""

from __future__ import annotations

from dataclasses import dataclass

from opti_med.medication_semantics.contracts import (
    MedicationClassAssignment,
    MedicationIdentityResolution,
)


FIRST_SCOPE_SUPPORTED_CLASSES: tuple[str, ...] = (
    "benzodiazepine",
    "opioid",
    "anticholinergic",
    "ppi",
    "antipsychotic",
)
FIRST_SCOPE_CLASS_SYSTEM = "opti_med_first_scope"
FIRST_SCOPE_CLASS_SYSTEM_VERSION = "2026.03_first_pass"

FIRST_SCOPE_INGREDIENT_CLASS_LABELS: dict[str, tuple[str, ...]] = {
    "alprazolam": ("benzodiazepine",),
    "aripiprazole": ("antipsychotic",),
    "benztropine": ("anticholinergic",),
    "chlorpromazine": ("antipsychotic",),
    "clonazepam": ("benzodiazepine",),
    "clozapine": ("antipsychotic",),
    "codeine": ("opioid",),
    "dexlansoprazole": ("ppi",),
    "diazepam": ("benzodiazepine",),
    "diphenhydramine": ("anticholinergic",),
    "esomeprazole": ("ppi",),
    "fentanyl": ("opioid",),
    "haloperidol": ("antipsychotic",),
    "hydrocodone": ("opioid",),
    "hydromorphone": ("opioid",),
    "hydroxyzine": ("anticholinergic",),
    "lansoprazole": ("ppi",),
    "lorazepam": ("benzodiazepine",),
    "methadone": ("opioid",),
    "midazolam": ("benzodiazepine",),
    "morphine": ("opioid",),
    "olanzapine": ("antipsychotic",),
    "omeprazole": ("ppi",),
    "oxycodone": ("opioid",),
    "oxybutynin": ("anticholinergic",),
    "pantoprazole": ("ppi",),
    "promethazine": ("anticholinergic",),
    "quetiapine": ("antipsychotic",),
    "rabeprazole": ("ppi",),
    "risperidone": ("antipsychotic",),
    "scopolamine": ("anticholinergic",),
    "temazepam": ("benzodiazepine",),
    "tolterodine": ("anticholinergic",),
    "tramadol": ("opioid",),
    "ziprasidone": ("antipsychotic",),
}


@dataclass(frozen=True, slots=True)
class FirstScopeRxNormClassAssigner:
    """Assign the initial ML-scope classes from RxNorm-standardized ingredients."""

    class_system: str = FIRST_SCOPE_CLASS_SYSTEM
    class_system_version: str = FIRST_SCOPE_CLASS_SYSTEM_VERSION

    def assign(
        self,
        resolution: MedicationIdentityResolution,
    ) -> MedicationClassAssignment:
        ingredient_id = _standardized_ingredient_id(resolution)
        ingredient_label = _standardized_ingredient_label(resolution)
        if ingredient_label is None:
            return MedicationClassAssignment(
                standardized_ingredient_id=ingredient_id,
                standardized_ingredient_label=None,
                assignment_status="no_standardized_ingredient_available",
                provenance={
                    "class_system": self.class_system,
                    "class_system_version": self.class_system_version,
                    "scope_classes": ",".join(FIRST_SCOPE_SUPPORTED_CLASSES),
                    "scope_reason": "rxnorm_ingredient_identity_missing",
                },
            )

        class_labels = FIRST_SCOPE_INGREDIENT_CLASS_LABELS.get(ingredient_label, tuple())
        if class_labels:
            return MedicationClassAssignment(
                standardized_ingredient_id=ingredient_id,
                standardized_ingredient_label=ingredient_label,
                class_ids=tuple(
                    f"{self.class_system}:{class_label}" for class_label in class_labels
                ),
                class_labels=class_labels,
                assignment_status="supported_scope_class_assigned",
                provenance={
                    "class_system": self.class_system,
                    "class_system_version": self.class_system_version,
                    "scope_classes": ",".join(FIRST_SCOPE_SUPPORTED_CLASSES),
                    "scope_reason": "ingredient_mapped_to_initial_supported_scope",
                },
            )

        return MedicationClassAssignment(
            standardized_ingredient_id=ingredient_id,
            standardized_ingredient_label=ingredient_label,
            assignment_status="supported_scope_unresolved",
            provenance={
                "class_system": self.class_system,
                "class_system_version": self.class_system_version,
                "scope_classes": ",".join(FIRST_SCOPE_SUPPORTED_CLASSES),
                "scope_reason": "ingredient_not_in_initial_supported_scope",
            },
        )


def _standardized_ingredient_id(
    resolution: MedicationIdentityResolution,
) -> str | None:
    ingredient_id = _normalized_text(resolution.ingredient_rxcui)
    if ingredient_id is not None:
        return ingredient_id
    if resolution.medication_standardized_source == "rxnorm_ingredient":
        return _normalized_text(resolution.returned_rxcui)
    return None


def _standardized_ingredient_label(
    resolution: MedicationIdentityResolution,
) -> str | None:
    ingredient_label = _normalized_text(resolution.ingredient_standardized)
    if ingredient_label is not None:
        return ingredient_label
    if resolution.medication_standardized_source == "rxnorm_ingredient":
        return _normalized_text(resolution.medication_standardized)
    return None


def _normalized_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None
