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
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


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
        medication_events: pd.DataFrame,
        existing_cache: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build the persisted mapping artifact using cache-first lookup behavior."""
        validate_medication_events_artifact(medication_events)
        queries = extract_medication_rxnorm_queries(medication_events)
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
            if cached_row is not None:
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
                raw_name=query["raw_medication_string"],
                normalized_hint=query["normalized_query_string"],
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


def extract_medication_rxnorm_queries(medication_events: pd.DataFrame) -> pd.DataFrame:
    """Extract one representative query row per medication candidate key."""
    if medication_events.empty:
        return pd.DataFrame(
            columns=[
                "medication_query_key",
                "raw_medication_string",
                "normalized_query_string",
            ]
        )

    working = medication_events.copy()
    working["medication_query_key"] = working.apply(_candidate_query_key, axis=1)
    working["raw_medication_string"] = working.apply(_representative_raw_text, axis=1)
    working["normalized_query_string"] = working["medication_normalized"].astype("string")
    working["_raw_present"] = working["raw_medication_string"].notna().astype(int)
    working = working.loc[working["medication_query_key"].notna()].copy()
    if working.empty:
        return pd.DataFrame(
            columns=[
                "medication_query_key",
                "raw_medication_string",
                "normalized_query_string",
            ]
        )
    queries = (
        working.sort_values(
            ["_raw_present", "medication_query_key", "raw_medication_string"],
            ascending=[False, True, True],
            na_position="last",
        )
        .drop_duplicates(subset=["medication_query_key"], keep="first")
        .loc[:, ["medication_query_key", "raw_medication_string", "normalized_query_string"]]
        .reset_index(drop=True)
    )
    queries["normalized_query_string"] = queries["normalized_query_string"].fillna(
        queries["medication_query_key"]
    )
    return queries


def _candidate_query_key(row: pd.Series) -> str | None:
    for value in [row.get("medication_standardized"), row.get("medication_normalized")]:
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
    row["raw_medication_string"] = (
        query.get("raw_medication_string") or cached_row.get("raw_medication_string")
    )
    row["normalized_query_string"] = (
        query.get("normalized_query_string") or cached_row.get("normalized_query_string")
    )
    return row


def _cache_only_miss_row(
    *,
    query: dict[str, object],
    mapper: RxNormBackedMedicationSemanticMapper,
    build_run_id: str,
) -> dict[str, object]:
    resolution = MedicationIdentityResolution(
        raw_name=_string_or_none(query.get("raw_medication_string")),
        normalized_query=_string_or_none(query.get("normalized_query_string")),
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
    if rxnorm_version is None and api_called_flag == 1:
        try:
            rxnorm_version = mapper.resolver.client.get_rxnorm_version()
        except Exception:
            rxnorm_version = None
    return {
        "medication_query_key": query["medication_query_key"],
        "raw_medication_string": _string_or_none(query.get("raw_medication_string")),
        "normalized_query_string": _string_or_none(query.get("normalized_query_string"))
        or _string_or_none(query.get("medication_query_key")),
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
