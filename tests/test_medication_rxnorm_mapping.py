"""Tests for the RxNorm mapper, cache builder, and state-table enrichment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import (
    MEDICATION_EVENT_COLUMNS,
    MEDICATION_EVENTS_CONTRACT_VERSION,
    MEDICATION_RXNORM_MAPPING_COLUMNS,
    MEDICATION_RXNORM_MAPPING_CONTRACT_VERSION,
)
from opti_med.data_access.encounter_medication_state import build_encounter_medication_state
from opti_med.data_access.medication_rxnorm_mapping import (
    LOOKUP_MODE_CACHE_FIRST,
    LOOKUP_MODE_CACHE_ONLY,
    MedicationRxNormMappingBuilder,
)
from opti_med.medication_semantics.contracts import MedicationIdentityResolution
from opti_med.medication_semantics.rxnorm import (
    PlaceholderMedicationClassAssigner,
    RxNormApiClient,
    RxNormBackedMedicationSemanticMapper,
    RxNormIdentityResolver,
)


class RxNormResolverTests(unittest.TestCase):
    def test_exact_lookup_resolves_to_related_ingredient(self) -> None:
        client = RxNormApiClient(transport=_fake_exact_transport)
        resolver = RxNormIdentityResolver(
            client=client,
            class_assigner=PlaceholderMedicationClassAssigner(),
        )

        resolution = resolver.resolve(raw_name="Lasix 20 MG Oral Tablet", normalized_hint="lasix")

        self.assertEqual(resolution.returned_rxcui, "123")
        self.assertEqual(resolution.matched_term, "Lasix 20 MG Oral Tablet")
        self.assertEqual(resolution.ingredient_standardized, "Furosemide")
        self.assertEqual(resolution.ingredient_resolution_status, "resolved_via_related_concept")
        self.assertEqual(resolution.medication_standardized_source, "rxnorm_ingredient")
        self.assertEqual(resolution.mapping_confidence, "high")

    def test_approximate_lookup_only_runs_after_exact_miss(self) -> None:
        client = RxNormApiClient(transport=_fake_approximate_transport)
        resolver = RxNormIdentityResolver(client=client)

        resolution = resolver.resolve(raw_name="Tylenol 325 mg", normalized_hint="tylenol")

        self.assertEqual(resolution.lookup_strategy_used, "approximate_fallback")
        self.assertEqual(resolution.returned_rxcui, "555")
        self.assertEqual(resolution.ingredient_standardized, "Acetaminophen")
        self.assertEqual(resolution.mapping_confidence, "medium")


class MedicationRxNormMappingBuilderTests(unittest.TestCase):
    def test_cache_hit_reuses_existing_mapping_without_api_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(analytical_root=Path(temp_dir) / "analytical")
            mapper = _CountingMapper()
            builder = MedicationRxNormMappingBuilder(
                settings,
                semantic_mapper=mapper,
                lookup_mode=LOOKUP_MODE_CACHE_FIRST,
            )
            medication_events = _minimal_medication_events(
                medication_normalized="furosemide",
                raw_medication_name="Furosemide 20 mg",
            )
            existing_cache = pd.DataFrame([_mapping_row("furosemide")])

            mapping = builder.build(
                medication_events=medication_events,
                existing_cache=existing_cache,
            )

            self.assertEqual(mapper.call_count, 0)
            self.assertEqual(mapping.iloc[0]["lookup_status"], "resolved")
            self.assertEqual(mapping.iloc[0]["medication_standardized"], "furosemide")

    def test_cache_only_mode_emits_explicit_unresolved_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(analytical_root=Path(temp_dir) / "analytical")
            mapper = _CountingMapper()
            builder = MedicationRxNormMappingBuilder(
                settings,
                semantic_mapper=mapper,
                lookup_mode=LOOKUP_MODE_CACHE_ONLY,
            )
            medication_events = _minimal_medication_events(
                medication_normalized="unknownmed",
                raw_medication_name="UnknownMed 10 mg",
            )

            mapping = builder.build(medication_events=medication_events)

            self.assertEqual(mapper.call_count, 0)
            self.assertEqual(mapping.iloc[0]["lookup_status"], "unresolved_cache_only_miss")
            self.assertEqual(
                mapping.iloc[0]["ingredient_resolution_status"],
                "unresolved_cache_only_miss",
            )
            self.assertEqual(mapping.iloc[0]["api_called_flag"], 0)

    def test_state_builder_groups_by_rxnorm_standardized_key_when_mapping_is_available(self) -> None:
        encounter_index = pd.DataFrame(
            [
                {
                    "subject_id": 1,
                    "encounter_id": "hadm:10",
                    "hadm_id": 10,
                    "stay_id": None,
                    "encounter_start": "2125-03-19 18:00:00",
                    "encounter_end": "2125-03-20 10:00:00",
                    "admittime": "2125-03-19 18:00:00",
                    "dischtime": "2125-03-20 10:00:00",
                    "intime": None,
                    "outtime": None,
                }
            ]
        )
        medication_events = pd.concat(
            [
                _minimal_medication_events(
                    medication_normalized="lasix",
                    raw_medication_name="Lasix 20 mg",
                    medication_event_id="event-1",
                    starttime="2125-03-19 19:00:00",
                    stoptime="2125-03-20 12:00:00",
                ),
                _minimal_medication_events(
                    medication_normalized="furosemide",
                    raw_medication_name="Furosemide 20 mg",
                    medication_event_id="event-2",
                    starttime="2125-03-19 20:00:00",
                    stoptime="2125-03-20 12:00:00",
                ),
            ],
            ignore_index=True,
        )
        mapping = pd.DataFrame(
            [
                _mapping_row(
                    "lasix",
                    raw_medication_string="Lasix 20 mg",
                    medication_standardized="Furosemide",
                    ingredient_standardized="Furosemide",
                    ingredient_resolution_status="resolved_via_related_concept",
                    medication_standardized_source="rxnorm_ingredient",
                    lookup_strategy_used="exact_or_normalized_raw_query",
                    rxnorm_rxcui="123",
                    matched_term="Lasix 20 MG Oral Tablet",
                ),
                _mapping_row(
                    "furosemide",
                    raw_medication_string="Furosemide 20 mg",
                    medication_standardized="Furosemide",
                    ingredient_standardized="Furosemide",
                    ingredient_resolution_status="resolved_to_ingredient",
                    medication_standardized_source="rxnorm_ingredient",
                    lookup_strategy_used="exact_or_normalized_normalized_query",
                    rxnorm_rxcui="321",
                    matched_term="Furosemide",
                    matched_term_type="IN",
                ),
            ]
        )

        state = build_encounter_medication_state(
            encounter_index=encounter_index,
            medication_events=medication_events,
            medication_rxnorm_mapping=mapping,
            review_time_policy="discharge_capped_latest_available",
        )

        self.assertEqual(len(state), 1)
        self.assertEqual(state.iloc[0]["medication_standardized"], "Furosemide")
        self.assertEqual(state.iloc[0]["medication_standardized_source"], "rxnorm_ingredient")
        self.assertEqual(state.iloc[0]["ingredient_standardized"], "Furosemide")
        self.assertEqual(state.iloc[0]["mapping_candidate_count"], 2)
        self.assertEqual(state.iloc[0]["active_at_review_flag"], 1)


class _CountingMapper:
    mapper_name = "fake-counting-mapper"
    mapper_version = "0.1"

    def __init__(self) -> None:
        self.call_count = 0
        self.class_assigner = PlaceholderMedicationClassAssigner()
        self.resolver = self

    def resolve(self, *, raw_name: object, normalized_hint: object = None) -> MedicationIdentityResolution:
        self.call_count += 1
        query = str(normalized_hint or raw_name)
        return MedicationIdentityResolution(
            raw_name=str(raw_name),
            normalized_query=query,
            lookup_strategy_used="fake_lookup",
            returned_rxcui="111",
            matched_term=query,
            matched_term_type="IN",
            medication_standardized=query,
            medication_standardized_source="rxnorm_ingredient",
            ingredient_rxcui="111",
            ingredient_standardized=query,
            ingredient_resolution_status="resolved_to_ingredient",
            mapping_confidence="high",
            provenance={},
        )


def _minimal_medication_events(
    *,
    medication_normalized: str,
    raw_medication_name: str,
    medication_event_id: str = "event-1",
    starttime: str = "2125-03-19 19:00:00",
    stoptime: str = "2125-03-20 12:00:00",
) -> pd.DataFrame:
    row = {
        "subject_id": 1,
        "hadm_id": 10,
        "stay_id": None,
        "encounter_id": "hadm:10",
        "encounter_source": "hospital_only",
        "encounter_start": "2125-03-19 18:00:00",
        "encounter_end": "2125-03-20 10:00:00",
        "medication_event_id": medication_event_id,
        "medication_event_type": "hospital_order",
        "event_source_category": "hospital_medication_order",
        "event_source_table": "prescriptions",
        "raw_medication_name": raw_medication_name,
        "medication_name": raw_medication_name,
        "medication_normalized": medication_normalized,
        "medication_prestandardized_text": medication_normalized,
        "event_time": starttime,
        "starttime": starttime,
        "stoptime": stoptime,
        "route": "PO",
        "frequency": "BID",
        "status": "active",
        "dose_value": "20",
        "dose_unit": "mg",
        "pharmacy_id": 500,
        "poe_id": 700,
        "emar_id": None,
        "emar_seq": None,
        "source_home_medrecon": 0,
        "source_ed_pyxis": 0,
        "source_hospital_order": 1,
        "source_hospital_admin": 0,
        "pharmacy_enriched_flag": 1,
        "order_enrichment_applied_flag": 1,
        "order_enrichment_source_table": "pharmacy",
        "continued_from_home_inferred": 0,
        "continued_from_home_inferred_flag": 0,
        "newly_started_during_encounter_inferred": 1,
        "newly_started_during_encounter_inferred_flag": 1,
        "continuity_inference_rule": None,
        "medication_episode_id": "episode-1",
        "prescription_segment_count": 1,
        "prescription_segments_json": "[]",
        "source_tables_json": "[\"prescriptions\"]",
        "source_record_provenance_json": "{\"role\": \"prescriptions\"}",
        "medication_event_build_run_id": "medication-events-test",
        "medication_event_contract_version": MEDICATION_EVENTS_CONTRACT_VERSION,
    }
    dataframe = pd.DataFrame([row])
    return dataframe.loc[:, MEDICATION_EVENT_COLUMNS]


def _mapping_row(
    medication_query_key: str,
    *,
    raw_medication_string: str | None = None,
    medication_standardized: str | None = None,
    ingredient_standardized: str | None = None,
    ingredient_resolution_status: str = "resolved_to_ingredient",
    medication_standardized_source: str = "rxnorm_ingredient",
    lookup_strategy_used: str = "cached_prior_result",
    rxnorm_rxcui: str | None = "111",
    matched_term: str | None = None,
    matched_term_type: str | None = "IN",
) -> dict[str, object]:
    row = {
        "medication_query_key": medication_query_key,
        "raw_medication_string": raw_medication_string or medication_query_key,
        "normalized_query_string": medication_query_key,
        "lookup_mode": LOOKUP_MODE_CACHE_FIRST,
        "lookup_strategy_used": lookup_strategy_used,
        "lookup_status": "resolved"
        if ingredient_resolution_status in {"resolved_to_ingredient", "resolved_via_related_concept"}
        else ingredient_resolution_status,
        "rxnorm_rxcui": rxnorm_rxcui,
        "matched_term": matched_term or medication_standardized or medication_query_key,
        "matched_term_type": matched_term_type,
        "ingredient_rxcui": rxnorm_rxcui,
        "ingredient_standardized": ingredient_standardized,
        "ingredient_resolution_status": ingredient_resolution_status,
        "medication_standardized": medication_standardized or medication_query_key,
        "medication_standardized_source": medication_standardized_source,
        "mapping_confidence": "high"
        if ingredient_resolution_status in {"resolved_to_ingredient", "resolved_via_related_concept"}
        else "none",
        "ambiguous_match_flag": 0,
        "candidate_match_count": 1,
        "candidate_rxcuis_json": "[\"111\"]" if rxnorm_rxcui else "[]",
        "ambiguity_note": None,
        "class_assignment_status": "class_assignment_pending",
        "class_ids_json": "[]",
        "class_labels_json": "[]",
        "lookup_timestamp": "2125-03-20 10:00:00",
        "api_called_flag": 1,
        "mapper_name": "rxnorm_api_bootstrap_mapper",
        "mapper_version": "0.1",
        "rxnorm_version": "2026-03-02",
        "rxnorm_api_version": "rxnav_rest_v1",
        "mapping_provenance_json": "{\"lookup_mode\": \"cache_first\"}",
        "medication_rxnorm_mapping_build_run_id": "medication-rxnorm-mapping-test",
        "medication_rxnorm_mapping_contract_version": MEDICATION_RXNORM_MAPPING_CONTRACT_VERSION,
    }
    return {column_name: row.get(column_name) for column_name in MEDICATION_RXNORM_MAPPING_COLUMNS}


def _fake_exact_transport(url: str, timeout_seconds: float) -> dict[str, object]:
    del timeout_seconds
    if "rxcui.json" in url and "Lasix" in url:
        return {"idGroup": {"rxnormId": ["123"]}}
    if "rxcui/123/properties.json" in url:
        return {"properties": {"name": "Lasix 20 MG Oral Tablet", "tty": "SCD"}}
    if "rxcui/123/related.json" in url:
        return {
            "relatedGroup": {
                "conceptGroup": [
                    {
                        "tty": "IN",
                        "conceptProperties": [
                            {"rxcui": "321", "name": "Furosemide", "tty": "IN"}
                        ],
                    }
                ]
            }
        }
    if "version.json" in url:
        return {"version": "2026-03-02"}
    raise AssertionError(f"Unexpected URL: {url}")


def _fake_approximate_transport(url: str, timeout_seconds: float) -> dict[str, object]:
    del timeout_seconds
    if "rxcui.json" in url:
        return {"idGroup": {}}
    if "approximateTerm.json" in url:
        return {"approximateGroup": {"candidate": [{"rxcui": "555", "rank": "1"}]}}
    if "rxcui/555/properties.json" in url:
        return {"properties": {"name": "Acetaminophen 325 MG Oral Tablet", "tty": "SCD"}}
    if "rxcui/555/related.json" in url:
        return {
            "relatedGroup": {
                "conceptGroup": [
                    {
                        "tty": "IN",
                        "conceptProperties": [
                            {"rxcui": "777", "name": "Acetaminophen", "tty": "IN"}
                        ],
                    }
                ]
            }
        }
    if "version.json" in url:
        return {"version": "2026-03-02"}
    raise AssertionError(f"Unexpected URL: {url}")


if __name__ == "__main__":
    unittest.main()
