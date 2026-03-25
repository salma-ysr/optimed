"""Decoupled RxNorm join step for the persisted 65+ encounter-medication-state artifact."""

from __future__ import annotations

from uuid import uuid4

import pandas as pd

from opti_med.data_access.artifact_schemas import (
    ENCOUNTER_MEDICATION_STATE_COLUMNS,
    ENCOUNTER_MEDICATION_STATE_CONTRACT_VERSION,
    validate_encounter_medication_state_artifact,
    validate_medication_rxnorm_mapping_artifact,
)
from opti_med.data_access.medication_rxnorm_mapping import (
    extract_medication_rxnorm_candidate_rows_from_encounter_medication_state,
)
from opti_med.data_access.provenance import loads_json_or_none


def build_encounter_medication_state_rxnorm(
    *,
    encounter_medication_state: pd.DataFrame,
    medication_rxnorm_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Join the persisted RxNorm mapping artifact back into the persisted 65+ state artifact."""
    validate_encounter_medication_state_artifact(encounter_medication_state)
    validate_medication_rxnorm_mapping_artifact(medication_rxnorm_mapping)
    if encounter_medication_state.empty:
        return pd.DataFrame(columns=ENCOUNTER_MEDICATION_STATE_COLUMNS)

    working = encounter_medication_state.reset_index(drop=True).copy()
    candidate_rows = extract_medication_rxnorm_candidate_rows_from_encounter_medication_state(
        working
    )
    working["_medication_query_key"] = candidate_rows["medication_query_key"]
    mapping = medication_rxnorm_mapping.loc[
        :,
        [
            "medication_query_key",
            "medication_standardized",
            "medication_standardized_source",
            "rxnorm_rxcui",
            "matched_term",
            "matched_term_type",
            "ingredient_standardized",
            "ingredient_resolution_status",
            "mapping_confidence",
            "ambiguous_match_flag",
            "candidate_match_count",
            "lookup_strategy_used",
            "class_labels_json",
        ],
    ].copy()
    mapping = mapping.rename(
        columns={
            "medication_query_key": "_medication_query_key",
            "medication_standardized": "_resolved_medication_standardized",
            "medication_standardized_source": "_resolved_medication_standardized_source",
            "rxnorm_rxcui": "_resolved_rxnorm_rxcui",
            "matched_term": "_resolved_rxnorm_matched_term",
            "matched_term_type": "_resolved_rxnorm_term_type",
            "ingredient_standardized": "_resolved_ingredient_standardized",
            "ingredient_resolution_status": "_resolved_ingredient_resolution_status",
            "mapping_confidence": "_resolved_mapping_confidence",
            "ambiguous_match_flag": "_resolved_ambiguous_mapping_flag",
            "candidate_match_count": "_resolved_candidate_match_count",
            "lookup_strategy_used": "_resolved_lookup_strategy_used",
            "class_labels_json": "_resolved_class_labels_json",
        }
    )
    enriched = working.merge(
        mapping,
        how="left",
        on="_medication_query_key",
        validate="many_to_one",
    )
    enriched = _apply_state_rxnorm_defaults(enriched)
    enriched["encounter_medication_state_build_run_id"] = (
        f"encounter-medication-state-rxnorm-{uuid4().hex[:12]}"
    )
    enriched["encounter_medication_state_contract_version"] = (
        ENCOUNTER_MEDICATION_STATE_CONTRACT_VERSION
    )
    enriched = enriched.loc[:, ENCOUNTER_MEDICATION_STATE_COLUMNS].copy()
    validate_encounter_medication_state_artifact(enriched)
    return enriched


def summarize_encounter_medication_state_rxnorm(
    encounter_medication_state_rxnorm: pd.DataFrame,
) -> list[str]:
    """Return compact summary lines for the decoupled RxNorm-joined state artifact."""
    validate_encounter_medication_state_artifact(encounter_medication_state_rxnorm)
    rows = len(encounter_medication_state_rxnorm)
    nonnull_rxcui = int(encounter_medication_state_rxnorm["rxnorm_rxcui"].notna().sum())
    nonnull_ingredient = int(
        encounter_medication_state_rxnorm["ingredient_standardized"].notna().sum()
    )
    unresolved = int(
        encounter_medication_state_rxnorm["ingredient_resolution_status"]
        .fillna("unresolved_no_match")
        .astype(str)
        .str.startswith("unresolved")
        .sum()
    )
    return [
        f"rows={rows:,}",
        f"rows_with_rxnorm_rxcui={nonnull_rxcui:,}",
        f"rows_with_ingredient_standardized={nonnull_ingredient:,}",
        f"rows_with_unresolved_ingredient_status={unresolved:,}",
    ]


def _apply_state_rxnorm_defaults(dataframe: pd.DataFrame) -> pd.DataFrame:
    enriched = dataframe.copy()
    query_keys = enriched["_medication_query_key"].apply(_string_or_none)
    existing_standardized = enriched["medication_standardized"].apply(_string_or_none)
    resolved_standardized = enriched["_resolved_medication_standardized"].apply(_string_or_none)
    existing_source = enriched["medication_standardized_source"].apply(_string_or_none)
    existing_rxcui = enriched["rxnorm_rxcui"].apply(_string_or_none)
    existing_matched_term = enriched["rxnorm_matched_term"].apply(_string_or_none)
    existing_term_type = enriched["rxnorm_term_type"].apply(_string_or_none)
    existing_ingredient = enriched["ingredient_standardized"].apply(_string_or_none)
    existing_status = enriched["ingredient_resolution_status"].apply(_string_or_none)
    existing_confidence = enriched["mapping_confidence"].apply(_string_or_none)
    resolved_source = enriched["_resolved_medication_standardized_source"].apply(_string_or_none)
    resolved_rxcui = enriched["_resolved_rxnorm_rxcui"].apply(_string_or_none)
    resolved_matched_term = enriched["_resolved_rxnorm_matched_term"].apply(_string_or_none)
    resolved_term_type = enriched["_resolved_rxnorm_term_type"].apply(_string_or_none)
    resolved_ingredient = enriched["_resolved_ingredient_standardized"].apply(_string_or_none)
    resolved_status = enriched["_resolved_ingredient_resolution_status"].apply(_string_or_none)
    resolved_confidence = enriched["_resolved_mapping_confidence"].apply(_string_or_none)
    use_mapping = _mapping_should_override_existing_semantics(
        existing_source=existing_source,
        existing_rxcui=existing_rxcui,
        existing_ingredient=existing_ingredient,
        resolved_status=resolved_status,
    )
    enriched["medication_standardized"] = resolved_standardized.where(
        use_mapping & resolved_standardized.notna(),
        existing_standardized,
    ).fillna(query_keys)
    enriched["medication_standardized_source"] = resolved_source.where(
        use_mapping & resolved_source.notna(),
        existing_source,
    ).fillna("normalized_text_fallback")
    enriched["rxnorm_rxcui"] = resolved_rxcui.where(
        use_mapping & resolved_rxcui.notna(),
        existing_rxcui,
    )
    enriched["rxnorm_matched_term"] = resolved_matched_term.where(
        use_mapping & resolved_matched_term.notna(),
        existing_matched_term,
    )
    enriched["rxnorm_term_type"] = resolved_term_type.where(
        use_mapping & resolved_term_type.notna(),
        existing_term_type,
    )
    enriched["ingredient_standardized"] = resolved_ingredient.where(
        use_mapping & resolved_ingredient.notna(),
        existing_ingredient,
    )
    enriched["ingredient_resolution_status"] = resolved_status.where(
        use_mapping & resolved_status.notna(),
        existing_status,
    ).fillna("unresolved_no_match")
    enriched["mapping_confidence"] = resolved_confidence.where(
        use_mapping & resolved_confidence.notna(),
        existing_confidence,
    ).fillna("none")
    existing_ambiguous = pd.to_numeric(
        enriched["ambiguous_mapping_flag"],
        errors="coerce",
    ).fillna(0)
    resolved_ambiguous = pd.to_numeric(
        enriched["_resolved_ambiguous_mapping_flag"],
        errors="coerce",
    ).fillna(0)
    enriched["ambiguous_mapping_flag"] = (
        pd.concat([existing_ambiguous, resolved_ambiguous], axis=1).max(axis=1).astype(int)
    )
    existing_candidate_count = pd.to_numeric(
        enriched["mapping_candidate_count"],
        errors="coerce",
    ).fillna(0)
    resolved_candidate_count = pd.to_numeric(
        enriched["_resolved_candidate_match_count"],
        errors="coerce",
    ).fillna(0)
    query_candidate_count = query_keys.notna().astype(int)
    enriched["mapping_candidate_count"] = (
        pd.concat(
            [existing_candidate_count, resolved_candidate_count, query_candidate_count],
            axis=1,
        )
        .max(axis=1)
        .astype(int)
    )
    enriched["medication_mapping_lookup_strategy"] = (
        enriched["_resolved_lookup_strategy_used"]
        .apply(_string_or_none)
        .where(
            use_mapping,
            enriched["medication_mapping_lookup_strategy"].apply(_string_or_none),
        )
        .fillna(enriched["medication_mapping_lookup_strategy"].apply(_string_or_none))
        .fillna("state_join_missing_mapping")
    )
    resolved_classes = enriched["_resolved_class_labels_json"].apply(
        _class_label_string_from_json
    )
    enriched["medication_class_standardized"] = resolved_classes.where(
        use_mapping & resolved_classes.notna(),
        enriched["medication_class_standardized"].apply(_string_or_none),
    ).fillna("unresolved")
    return enriched


def _class_label_string_from_json(value: object) -> str | None:
    payload = loads_json_or_none(value)
    if not isinstance(payload, list):
        return None
    labels = sorted(
        {
            str(item).strip()
            for item in payload
            if str(item).strip()
        }
    )
    if not labels:
        return "unresolved"
    return "|".join(labels)


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    return text or None


def _mapping_should_override_existing_semantics(
    *,
    existing_source: pd.Series,
    existing_rxcui: pd.Series,
    existing_ingredient: pd.Series,
    resolved_status: pd.Series,
) -> pd.Series:
    existing_has_enrichment = (
        existing_source.isin({"rxnorm_ingredient", "rxnorm_term"})
        | existing_rxcui.notna()
        | existing_ingredient.notna()
    )
    mapping_has_resolved_semantics = resolved_status.isin(
        {
            "resolved_to_ingredient",
            "resolved_via_related_concept",
            "resolved_term_only_no_ingredient",
        }
    )
    return mapping_has_resolved_semantics | ~existing_has_enrichment
