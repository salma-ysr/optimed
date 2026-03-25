"""Persistent RxNorm mapping artifact and cache-first builder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import (
    MEDICATION_RXNORM_MAPPING_COLUMNS,
    MEDICATION_RXNORM_MAPPING_CONTRACT_VERSION,
    validate_encounter_medication_state_artifact,
    validate_medication_events_artifact,
    validate_medication_rxnorm_mapping_artifact,
)
from opti_med.data_access.provenance import dumps_json
from opti_med.medication_semantics import RxNormBackedMedicationSemanticMapper
from opti_med.medication_semantics.contracts import MedicationIdentityResolution


MedicationRxNormLookupMode = str
LOOKUP_MODE_CACHE_FIRST = "cache_first"
LOOKUP_MODE_CACHE_ONLY = "cache_only"
LOOKUP_MODES = {LOOKUP_MODE_CACHE_FIRST, LOOKUP_MODE_CACHE_ONLY}
RETRYABLE_CACHE_STATUSES = {"unresolved_api_error", "unresolved_cache_only_miss"}
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
QUERY_COLUMNS = [
    "medication_query_key",
    "medication_raw",
    "medication_normalized",
    "raw_medication_string",
    "normalized_query_string",
]


@dataclass(frozen=True, slots=True)
class MedicationRxNormMappingBuildResult:
    """Built RxNorm mapping/cache artifact and its output path."""

    dataframe: pd.DataFrame
    output_path: Path


class MedicationRxNormMappingBuilder:
    """Build a persistent RxNorm mapping artifact from medication candidates."""

    def __init__(
        self,
        settings: Settings,
        *,
        semantic_mapper: RxNormBackedMedicationSemanticMapper | None = None,
        lookup_mode: MedicationRxNormLookupMode = LOOKUP_MODE_CACHE_FIRST,
    ) -> None:
        if lookup_mode not in LOOKUP_MODES:
            allowed = ", ".join(sorted(LOOKUP_MODES))
            raise ValueError(f"Unsupported RxNorm lookup mode '{lookup_mode}'. Expected one of: {allowed}.")
        self.settings = settings
        self.semantic_mapper = semantic_mapper or RxNormBackedMedicationSemanticMapper()
        self.lookup_mode = lookup_mode

    def build(
        self,
        *,
        medication_events: pd.DataFrame | None = None,
        mapping_queries: pd.DataFrame | None = None,
        existing_cache: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build the persisted mapping artifact using cache-first lookup behavior."""
        if (medication_events is None) == (mapping_queries is None):
            raise ValueError(
                "Provide exactly one of medication_events or mapping_queries when building "
                "medication_rxnorm_mapping."
            )
        if medication_events is not None:
            validate_medication_events_artifact(medication_events)
            queries = extract_medication_rxnorm_queries(medication_events)
        else:
            queries = normalize_medication_rxnorm_queries(mapping_queries)
        if queries.empty:
            return pd.DataFrame(columns=MEDICATION_RXNORM_MAPPING_COLUMNS)

        cache = existing_cache if existing_cache is not None else self._load_existing_cache()
        if cache is not None and not cache.empty:
            validate_medication_rxnorm_mapping_artifact(cache)
            cached_rows = {
                str(row["medication_query_key"]): row
                for row in cache.sort_values("lookup_timestamp", na_position="last").to_dict(orient="records")
            }
        else:
            cached_rows = {}

        build_run_id = f"medication-rxnorm-mapping-{uuid4().hex[:12]}"
        built_rows: list[dict[str, object]] = []
        for query in queries.to_dict(orient="records"):
            query_key = str(query["medication_query_key"])
            cached_row = cached_rows.get(query_key)
            if cached_row is not None and _should_reuse_cached_row(
                cached_row,
                lookup_mode=self.lookup_mode,
            ):
                built_rows.append(
                    _merge_cached_row_with_query(
                        cached_row=cached_row,
                        query=query,
                    )
                )
                continue

            if self.lookup_mode == LOOKUP_MODE_CACHE_ONLY:
                built_rows.append(
                    _cache_only_miss_row(
                        query=query,
                        mapper=self.semantic_mapper,
                        build_run_id=build_run_id,
                    )
                )
                continue

            resolution = self.semantic_mapper.resolver.resolve(
                raw_name=query["medication_raw"],
                normalized_hint=query["medication_normalized"],
            )
            built_rows.append(
                _resolution_to_mapping_row(
                    query=query,
                    resolution=resolution,
                    mapper=self.semantic_mapper,
                    lookup_mode=self.lookup_mode,
                    build_run_id=build_run_id,
                    api_called_flag=1,
                )
            )

        mapping = pd.DataFrame(built_rows)
        if mapping.empty:
            return pd.DataFrame(columns=MEDICATION_RXNORM_MAPPING_COLUMNS)
        mapping = (
            mapping.sort_values(["medication_query_key", "lookup_timestamp"], na_position="last")
            .drop_duplicates(subset=["medication_query_key"], keep="last")
            .reset_index(drop=True)
        )
        mapping = mapping.loc[:, MEDICATION_RXNORM_MAPPING_COLUMNS].copy()
        validate_medication_rxnorm_mapping_artifact(mapping)
        return mapping

    def save(
        self,
        dataframe: pd.DataFrame,
        output_path: Path | None = None,
    ) -> MedicationRxNormMappingBuildResult:
        """Persist the medication RxNorm mapping/cache artifact to Parquet."""
        validate_medication_rxnorm_mapping_artifact(dataframe)
        target_path = output_path or self.settings.medication_rxnorm_mapping_output_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(target_path, index=False)
        return MedicationRxNormMappingBuildResult(
            dataframe=dataframe,
            output_path=target_path,
        )

    def _load_existing_cache(self) -> pd.DataFrame | None:
        path = self.settings.medication_rxnorm_mapping_output_path
        if not path.exists():
            return None
        return pd.read_parquet(path)


def _should_reuse_cached_row(
    cached_row: dict[str, object],
    *,
    lookup_mode: str,
) -> bool:
    if lookup_mode == LOOKUP_MODE_CACHE_ONLY:
        return True
    status = _string_or_none(cached_row.get("lookup_status"))
    return status not in RETRYABLE_CACHE_STATUSES


def extract_medication_rxnorm_queries(medication_events: pd.DataFrame) -> pd.DataFrame:
    """Extract one representative query row per medication candidate key."""
    if medication_events.empty:
        return pd.DataFrame(columns=QUERY_COLUMNS)

    working = medication_events.copy()
    working["medication_query_key"] = working.apply(_event_candidate_query_key, axis=1)
    working["medication_raw"] = working.apply(_representative_raw_text, axis=1)
    working["medication_normalized"] = working["medication_normalized"].apply(_string_or_none)
    return normalize_medication_rxnorm_queries(
        working.loc[:, ["medication_query_key", "medication_raw", "medication_normalized"]]
    )


def extract_medication_rxnorm_candidate_rows_from_encounter_medication_state(
    encounter_medication_state: pd.DataFrame,
) -> pd.DataFrame:
    """Extract one RxNorm mapping candidate row per persisted 65+ state row."""
    validate_encounter_medication_state_artifact(encounter_medication_state)
    if encounter_medication_state.empty:
        return pd.DataFrame(columns=QUERY_COLUMNS)

    working = encounter_medication_state.copy()
    working["medication_raw"] = working["medication_raw"].apply(_string_or_none)
    working["medication_normalized"] = working["medication_normalized"].apply(_string_or_none)
    working["medication_query_key"] = working.apply(_state_candidate_query_key, axis=1)
    working = working.loc[working["medication_query_key"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=QUERY_COLUMNS)
    working["normalized_query_string"] = working["medication_normalized"].fillna(
        working["medication_query_key"]
    )
    working["medication_normalized"] = working["normalized_query_string"]
    working["medication_raw"] = working["medication_raw"].fillna(
        working["normalized_query_string"]
    )
    working["raw_medication_string"] = working["medication_raw"]
    return working.loc[:, QUERY_COLUMNS].reset_index(drop=True)


def extract_medication_rxnorm_queries_from_encounter_medication_state(
    encounter_medication_state: pd.DataFrame,
) -> pd.DataFrame:
    """Extract unique RxNorm mapping candidates from the persisted 65+ state artifact."""
    return normalize_medication_rxnorm_queries(
        extract_medication_rxnorm_candidate_rows_from_encounter_medication_state(
            encounter_medication_state
        )
    )


def normalize_medication_rxnorm_queries(mapping_queries: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize any candidate-query frame to the canonical mapping-query columns."""
    if mapping_queries is None or mapping_queries.empty:
        return pd.DataFrame(columns=QUERY_COLUMNS)

    working = mapping_queries.copy()
    if "medication_query_key" not in working:
        working["medication_query_key"] = pd.Series(pd.NA, index=working.index, dtype="object")
    if "medication_raw" not in working:
        working["medication_raw"] = working.get("raw_medication_string")
    if "medication_normalized" not in working:
        working["medication_normalized"] = working.get("normalized_query_string")

    working["medication_query_key"] = working["medication_query_key"].apply(_string_or_none)
    working["medication_raw"] = working["medication_raw"].apply(_string_or_none)
    working["medication_normalized"] = working["medication_normalized"].apply(_string_or_none)
    missing_query_key = working["medication_query_key"].isna()
    if missing_query_key.any():
        working.loc[missing_query_key, "medication_query_key"] = working.loc[
            missing_query_key
        ].apply(_query_key_from_query_row, axis=1)
    working = working.loc[working["medication_query_key"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=QUERY_COLUMNS)

    working["normalized_query_string"] = working["medication_normalized"].fillna(
        working["medication_query_key"]
    )
    working["medication_normalized"] = working["normalized_query_string"]
    working["medication_raw"] = working["medication_raw"].fillna(
        working["normalized_query_string"]
    )
    working["raw_medication_string"] = working["medication_raw"]
    working["_raw_present"] = working["medication_raw"].notna().astype(int)
    return (
        working.sort_values(
            ["_raw_present", "medication_query_key", "medication_raw"],
            ascending=[False, True, True],
            na_position="last",
        )
        .drop_duplicates(subset=["medication_query_key"], keep="first")
        .loc[:, QUERY_COLUMNS]
        .reset_index(drop=True)
    )


def _event_candidate_query_key(row: pd.Series) -> str | None:
    return _first_non_empty_text(
        [
            row.get("medication_normalized"),
            row.get("raw_medication_name"),
            row.get("medication_name"),
        ]
    )


def _state_candidate_query_key(row: pd.Series) -> str | None:
    return _first_non_empty_text(
        [
            row.get("medication_normalized"),
            row.get("medication_raw"),
        ]
    )


def _query_key_from_query_row(row: pd.Series) -> str | None:
    return _first_non_empty_text(
        [
            row.get("medication_normalized"),
            row.get("normalized_query_string"),
            row.get("medication_raw"),
            row.get("raw_medication_string"),
        ]
    )


def _representative_raw_text(row: pd.Series) -> str | None:
    for value in [row.get("raw_medication_name"), row.get("medication_name"), row.get("medication_normalized")]:
        if value is None:
            continue
        if value is pd.NA:
            continue
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        text = str(value).strip()
        if text:
            return text
    return None


def _merge_cached_row_with_query(
    *,
    cached_row: dict[str, object],
    query: dict[str, object],
) -> dict[str, object]:
    row = dict(cached_row)
    medication_raw = (
        query.get("medication_raw")
        or query.get("raw_medication_string")
        or cached_row.get("medication_raw")
        or cached_row.get("raw_medication_string")
    )
    medication_normalized = (
        query.get("medication_normalized")
        or query.get("normalized_query_string")
        or cached_row.get("medication_normalized")
        or cached_row.get("normalized_query_string")
    )
    row["medication_raw"] = medication_raw
    row["raw_medication_string"] = medication_raw
    row["medication_normalized"] = medication_normalized
    row["normalized_query_string"] = medication_normalized
    return row


def _cache_only_miss_row(
    *,
    query: dict[str, object],
    mapper: RxNormBackedMedicationSemanticMapper,
    build_run_id: str,
) -> dict[str, object]:
    resolution = MedicationIdentityResolution(
        raw_name=_string_or_none(query.get("medication_raw") or query.get("raw_medication_string")),
        normalized_query=_string_or_none(
            query.get("medication_normalized") or query.get("normalized_query_string")
        ),
        lookup_strategy_used="cache_only_miss",
        returned_rxcui=None,
        matched_term=None,
        matched_term_type=None,
        medication_standardized=_string_or_none(query.get("medication_query_key")),
        medication_standardized_source="normalized_text_fallback",
        ingredient_rxcui=None,
        ingredient_standardized=None,
        ingredient_resolution_status="unresolved_cache_only_miss",
        mapping_confidence="none",
        ambiguous_match_flag=False,
        multi_hit_candidate_count=0,
        candidate_rxcuis=tuple(),
        provenance={"ambiguity_note": "No cached mapping was available and API calls were disabled."},
    )
    return _resolution_to_mapping_row(
        query=query,
        resolution=resolution,
        mapper=mapper,
        lookup_mode=LOOKUP_MODE_CACHE_ONLY,
        build_run_id=build_run_id,
        api_called_flag=0,
    )


def _resolution_to_mapping_row(
    *,
    query: dict[str, object],
    resolution: MedicationIdentityResolution,
    mapper: RxNormBackedMedicationSemanticMapper,
    lookup_mode: str,
    build_run_id: str,
    api_called_flag: int,
) -> dict[str, object]:
    lookup_status = _lookup_status_from_resolution(resolution)
    class_assignment = mapper.class_assigner.assign(resolution)
    rxnorm_version = resolution.provenance.get("rxnorm_version")
    medication_raw = _string_or_none(query.get("medication_raw") or query.get("raw_medication_string"))
    medication_normalized = _string_or_none(
        query.get("medication_normalized") or query.get("normalized_query_string")
    ) or _string_or_none(query.get("medication_query_key"))
    unresolved_reason = _unresolved_reason_for_lookup_status(lookup_status)
    if rxnorm_version is None and api_called_flag == 1:
        try:
            rxnorm_version = mapper.resolver.client.get_rxnorm_version()
        except Exception:
            rxnorm_version = None
    return {
        "medication_query_key": query["medication_query_key"],
        "medication_raw": medication_raw or medication_normalized,
        "medication_normalized": medication_normalized,
        "raw_medication_string": medication_raw or medication_normalized,
        "normalized_query_string": medication_normalized,
        "lookup_mode": lookup_mode,
        "lookup_strategy_used": resolution.lookup_strategy_used,
        "lookup_status": lookup_status,
        "rxnorm_rxcui": resolution.returned_rxcui,
        "matched_term": resolution.matched_term,
        "matched_term_type": resolution.matched_term_type,
        "ingredient_rxcui": resolution.ingredient_rxcui,
        "ingredient_standardized": resolution.ingredient_standardized,
        "ingredient_resolution_status": resolution.ingredient_resolution_status,
        "medication_standardized": resolution.medication_standardized
        or _string_or_none(query.get("medication_query_key")),
        "medication_standardized_source": resolution.medication_standardized_source
        or "normalized_text_fallback",
        "mapping_confidence": resolution.mapping_confidence,
        "ambiguous_match_flag": int(resolution.ambiguous_match_flag),
        "candidate_match_count": int(resolution.multi_hit_candidate_count),
        "candidate_rxcuis_json": dumps_json(sorted(resolution.candidate_rxcuis)),
        "ambiguity_note": _string_or_none(resolution.provenance.get("ambiguity_note")),
        "unresolved_reason": unresolved_reason,
        "class_assignment_status": class_assignment.assignment_status,
        "class_ids_json": dumps_json(sorted(class_assignment.class_ids)),
        "class_labels_json": dumps_json(sorted(class_assignment.class_labels)),
        "lookup_timestamp": pd.Timestamp.utcnow().strftime(TIMESTAMP_FORMAT),
        "api_called_flag": int(api_called_flag),
        "mapper_name": mapper.mapper_name,
        "mapper_version": mapper.mapper_version,
        "rxnorm_version": _string_or_none(rxnorm_version),
        "rxnorm_api_version": "rxnav_rest_v1",
        "mapping_provenance_json": dumps_json(
            {
                "query_key": query["medication_query_key"],
                "raw_name": resolution.raw_name,
                "normalized_query": resolution.normalized_query,
                "class_assignment_status": class_assignment.assignment_status,
                "approximate_rank": resolution.provenance.get("approximate_rank"),
                "lookup_mode": lookup_mode,
            }
        ),
        "medication_rxnorm_mapping_build_run_id": build_run_id,
        "medication_rxnorm_mapping_contract_version": (
            MEDICATION_RXNORM_MAPPING_CONTRACT_VERSION
        ),
    }


def _lookup_status_from_resolution(resolution: MedicationIdentityResolution) -> str:
    status = resolution.ingredient_resolution_status
    if status in {"resolved_to_ingredient", "resolved_via_related_concept"}:
        return "resolved"
    if status == "resolved_term_only_no_ingredient":
        return "resolved_term_only_no_ingredient"
    return status


def _unresolved_reason_for_lookup_status(lookup_status: str) -> str | None:
    if lookup_status == "resolved":
        return None
    return lookup_status


def _first_non_empty_text(values: list[object]) -> str | None:
    for value in values:
        text = _string_or_none(value)
        if text is not None:
            return text
    return None


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
